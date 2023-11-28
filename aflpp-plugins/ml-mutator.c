// Copyright (c) 2023 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
/**
 * A Custom Mutator for AFL++ that implements Neuzz++.
 * Therefore, a neural network is trained with seed inputs and the corresponding edge coverage map.
 * The mutation is then guided by using gradient information from the neural network.
**/

// You need to use -I /path/to/AFLplusplus/include
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <math.h>
#include <assert.h>

#include "debug.h"
#include "alloc-inl.h"
#include "afl-fuzz.h"
#include "custom_mutator_helpers.h"




/*Minimum number of seeds to start ML */
#define INITIAL_TRAIN_THRESHOLD 200

/* The number of insert and delete operations that should be performed, as a ratio of the number of gradients provided (must be <= 1)*/
#define INS_DEL_OPERATIONS_RATIO 0.2

/* The buffer size to generate customized descriptions (names) for newly found seeds */
#define DESCRIBE_BUFFER_SIZE 100

/* The number of edges that should be targeted per seed (lines in gradfile) */
#define TARGET_EDGES_PER_SEED 5


/* Filename of the named pipe to the ml model */
#define NAMED_PIPE_OUT  "pipe_to_ml_model"

/* Filename of the named pipe from the ml model */
#define NAMED_PIPE_IN   "pipe_from_ml_model"

typedef struct {
    /**
     * Here goes all stuff that needs to be preserved (state of the plugin)
     * */
    afl_state_t* afl; /* The AFL State Object */
    BUF_VAR(int32_t, loc); /* The location array for current seed to fuzz ( comes from ML part) */
    BUF_VAR(int32_t, sign); /* The sign of the gradient in line with loc[] ( comes from ML part) */
    BUF_VAR(u8, up_steps); /* Up steps for each iteration (from NEUZZ) */
    BUF_VAR(u8, down_steps); /* Down steps for each iteration (from NEUZZ)  */
    uint32_t iter_cnt; /* In which iteration (range) of current seed are we?*/
    uint32_t ins_del_mut_cnt; /* How many ins/del mutations are left?*/
    BUF_VAR(u8, ml_mutator); /* Buffer to perform ins operations that do not fit into original buffer*/
    uint32_t num_of_gradients; /* The number of gradients in the gradfile for the current seed */
    uint32_t num_of_iterations; /* The number of needed iterations for the current seed = log2(num_of_gradients)*/
    uint32_t num_of_ins_del_operations; /* Number of additional ins and del mutations based on offsets from grad line*/
    char describe_buffer[DESCRIBE_BUFFER_SIZE];
    char* out_pipe_file_name; /* Filename of pipe to ml model */
    char* in_pipe_file_name; /* Filename of pipe from model */
    FILE* out_pipe_fd; /* File descriptor of pipe to ml model */
    FILE* in_pipe_fd; /* File descriptor of pipe from ml model */
    pid_t ml_pid; /* The process ID of the ml part */
    char *gradient_file_line; /* Pointer to head buffer for a gradient file line */
} ml_mutator_state_t;




/**
 * Starts the ML model process.
 * */
void start_ml_model_process(ml_mutator_state_t *data) {


  // Let the ML python script run in other process, so that we can change env variables and eventually let the training proceed in background
  data->ml_pid = fork();

  if(data->ml_pid == 0) { //child process

    //unset env variables from afl
    unsetenv("__AFL_OUT_DIR");
    unsetenv("__AFL_LOCKFILE");
    unsetenv("__AFL_PERSISTENT");
    unsetenv("__AFL_DEFER_FORKSRV");
    unsetenv("__AFL_SHM_FUZZ_ID");
    unsetenv("__AFL_SHM_ID");
    unsetenv("__AFL_TARGET_PID1");

    char* neuzzpp_path = getenv("NEUZZPP_PATH");
    if(!neuzzpp_path) { // if path is not set try working directory
      neuzzpp_path = ".";
    }
    setenv("PYTHONPATH", neuzzpp_path, 1);

    int argc = 0;
    char* argv[150]; // 150 arguments should be enough

    argv[argc++] = "python";

    argv[argc++] = (char *) alloc_printf("%s/scripts/train_cov_oracle.py", neuzzpp_path);
    
    argv[argc++] = "-f"; // Fast training: no checkpoints, only some ML metrics measured
    argv[argc++] = "-s"; // Add early stopping to avoid overfitting
    argv[argc++] = "10";

    //Here we interconnect the in and out pipe 
    argv[argc++] = data->out_pipe_file_name;
    argv[argc++] = data->in_pipe_file_name;

    argv[argc++] = (char *) alloc_printf("%s/queue/", data->afl->out_dir);

    /* Read the command line file */
    u8 *tmp = alloc_printf("%s/cmdline", data->afl->out_dir);
    FILE *cmdline_file = fopen((char*) tmp, "r");
    if (cmdline_file <= 0) { PFATAL("Unable to read '%s'", tmp); }
    ck_free(tmp);

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    // parse the command line file created from AFL and append to command
    while ((read = getline(&line, &len, cmdline_file)) != -1) {
      if (line[read - 1] == '\n') {
        line[read - 1] = '\0';
      }

      argv[argc++] = (char*) alloc_printf("%s", line); // not freed, but doesn't matter because of exec
    }

    argv[argc++] = NULL; // array delimiter

    if (line) {
      free(line);
    }

    fclose(cmdline_file);

    printf("\n\n\n\n\n Execute ML python script");
    for(int i = 0; i < argc; i++) {
      printf("%s ", argv[i]);
    }
    printf("\n");

    // Launch Python model training
    int retcode = execvp(argv[0], argv);
    FATAL("Executing the ML python script failed. Should never happen! errno: (%d), retcode: (%d)", errno, retcode);

    /*Here would go the free calls :) */

  } else { //parent process
    
    // this will block until the other end connects
    data->in_pipe_fd = fopen(data->in_pipe_file_name, "r");
    data->out_pipe_fd = fopen(data->out_pipe_file_name, "w");
    
    if(data->out_pipe_fd < 0 || data->in_pipe_fd < 0)
      perror("Opening FIFOs failed!"); 

  }

}


/**
 * Initialize this custom mutator
 *
 * @param[in] afl a pointer to the internal state object. Can be ignored for
 * now.
 * @param[in] seed A seed for this mutator - the same seed should always mutate
 * in the same way.
 * @return Pointer to the data object this custom mutator instance should use.
 *         There may be multiple instances of this mutator in one afl-fuzz run!
 *         Return NULL on error.
 */
ml_mutator_state_t *afl_custom_init(afl_state_t *afl, unsigned int seed) {

  srand(seed);  // needed also by surgical_havoc_mutate()

  ml_mutator_state_t *data = calloc(1, sizeof(ml_mutator_state_t));
  if (!data) {

    perror("afl_custom_init alloc");
    return NULL;

  }

  data->afl = afl;



  printf("Setup named pipes\n");

  // Setup named pipes for communication with ml part
  data->out_pipe_file_name = (char*) alloc_printf("%s/%s", data->afl->out_dir, NAMED_PIPE_OUT);
  data->in_pipe_file_name = (char*) alloc_printf("%s/%s", data->afl->out_dir, NAMED_PIPE_IN);

  struct stat sb;

  //Remove files/pipes if they already exist
  if(stat(data->out_pipe_file_name, &sb) == 0)
    unlink(data->out_pipe_file_name);
  
  if(stat(data->in_pipe_file_name, &sb) == 0)
    unlink(data->in_pipe_file_name);

  if(mkfifo(data->out_pipe_file_name, 0777) != 0)
    perror("Creating FIFO failed!");

  if(mkfifo(data->in_pipe_file_name, 0777) != 0)
    perror("Creating FIFO failed!"); 


  // Unfortunately, we can not initialize the ml part yet,
  // because AFL is not fully initialized at this point and
  // the cmdline file is not yet available

  return data;

}



/* parse one line of gradient string into array (max N elements) */
void parse_array(char * str, int *array, uint32_t N){

  uint32_t i=0;

  char* token = strtok(str,",");

  while(token != NULL && i < N){
    array[i]=atoi(token);
    i++;
    token = strtok(NULL, ",");
  }

  return;
}

/* count number of elements from gradient string*/
uint32_t count_elements(char * str){

  if(strlen(str) == 0 ) { // This should be the only condition for 'No element'
    return 0;
  }

  // start at one element, because separator is only used between elements and not at the beginning and the end.
  uint32_t i=1;


  char *ptr = str;
  while((ptr = strchr(ptr, ',')) != NULL) {
    i++;
    ptr++;
  }

  return i;
}


/**
 * This method is called just before fuzzing a queue entry with the custom
 * mutator, and receives the initial buffer. It should return the number of
 * fuzzes to perform.
 *
 * A value of 0 means no fuzzing of this queue entry.
 *
 * Here the machine learning is triggered, each new AFL++ cycle if at least <INITIAL_TRAIN_THRESHOLD> seeds are in corpus
 *
 * (Optional)
 *
 * @param data pointer returned in afl_custom_init by this custom mutator
 * @param buf Buffer containing the test case
 * @param buf_size Size of the test case
 * @return The amount of fuzzes to perform on this queue entry, 0 = skip
 */
uint32_t afl_custom_fuzz_count (ml_mutator_state_t *data, const u8 *buf, size_t buf_size) {


  if(data->afl->queue_cur->len > buf_size) {
    return 0; //Probably called from splice stage. We can not apply our mutations then :/
  }

  // Cannot use ML model until we have enough seeds 
  if(data->afl->queued_paths <= INITIAL_TRAIN_THRESHOLD) {
    return 0;
  }


  // check if ML part is not yet initialized
  if(!data->ml_pid) {
    start_ml_model_process(data);
  }

  /**
   * - Request gradient line for requested seed
   * - Parse gradient line for here
   * - Put it in memory for next mutation iterations
   * - Calculate and return number of desired mutations
   * */
  char *current_fname = strrchr((char*) data->afl->queue_cur->fname, '/');

  //send filename to ml part
  fputs(current_fname, data->out_pipe_fd);
  fputc('\n', data->out_pipe_fd);
  fflush(data->out_pipe_fd);



  size_t llen = 0;
  ssize_t nread;

  if ((nread = getline(&data->gradient_file_line, &llen, data->in_pipe_fd)) != -1) {
    /* parse gradient info */
    char* loc_str = strtok(data->gradient_file_line,"|");
    char* sign_str = strtok(NULL,"|");


    
    /* count elements and ensure buffer sizes */
    data->num_of_gradients = count_elements(loc_str); //number of sign should be equal!

    /*Eventually limit number of processed gradients*/
    if(getenv("NEUZZPP_MAX_GRADS")) {
      uint32_t max_num_of_gradients =  strtoul(getenv("NEUZZPP_MAX_GRADS"), NULL, 10);
      data->num_of_gradients = MIN(max_num_of_gradients, data->num_of_gradients);
    }
    data->num_of_iterations = (uint32_t) data->num_of_gradients == 1 ? 1 : log2(data->num_of_gradients);

    //Neuzz had fixed 2048 operations here, which does not work anymore (must be < num_of_gradients)!
    data->num_of_ins_del_operations = (uint32_t) (data->num_of_gradients * INS_DEL_OPERATIONS_RATIO);


    maybe_grow(BUF_PARAMS(data, loc), data->num_of_gradients * sizeof(int32_t));
    maybe_grow(BUF_PARAMS(data, sign), data->num_of_gradients* sizeof(int32_t));

    parse_array(loc_str,data->loc_buf, data->num_of_gradients);
    parse_array(sign_str,data->sign_buf, data->num_of_gradients);
  } else {
    data->num_of_gradients = 0;
    data->num_of_iterations = 0;
    data->num_of_ins_del_operations = 0;
  }



  //Initialize data structures
  //Allocate num_of_iterations +1 for simpler recognizing of end of mutations
  maybe_grow(BUF_PARAMS(data, up_steps), data->num_of_iterations + 1);
  maybe_grow(BUF_PARAMS(data, down_steps), data->num_of_iterations + 1);
  memset(data->up_steps_buf, 0, sizeof(data->up_steps_buf[0]) * (data->num_of_iterations + 1));
  memset(data->down_steps_buf, 0, sizeof(data->down_steps_buf[0]) * (data->num_of_iterations + 1));

  maybe_grow(BUF_PARAMS(data, ml_mutator), buf_size);


/**
 * Calculate number of up_steps + low_steps so that we can return that number to AFL before fuzzing the specific seed
 * Original NEUZZ does ranges from {0,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192};
 *
 **/

  uint32_t sum_steps = 0;
/* Copied from NEUZZ!! */
/* More adaptive implementation by max now. Do a iteration only if its size fits fully into the original seed*/
/* flip interesting locations */
  for(int iter=0 ;iter < data->num_of_iterations && pow(2, iter + 1) <= buf_size; iter++){

    /* find mutation range for every iteration */
    int low_index = iter == 0 ? 0 : pow(2, iter); // [14] = {0,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192};
    int up_index = pow(2, iter + 1);

    for(int index=low_index; index<up_index; index=index+1){
      u8 cur_up_step = 0;
      u8 cur_down_step = 0;
      if(data->sign_buf[index] == 1){
          cur_up_step = (255 - (u8)buf[data->loc_buf[index]]);
          if(cur_up_step > data->up_steps_buf[iter])
              data->up_steps_buf[iter] = cur_up_step;
          cur_down_step = (u8)(buf[data->loc_buf[index]]);
          if(cur_down_step > data->down_steps_buf[iter])
              data->down_steps_buf[iter] = cur_down_step;
      }
      else{
          cur_up_step = (u8)buf[data->loc_buf[index]];
          if(cur_up_step > data->up_steps_buf[iter])
              data->up_steps_buf[iter] = cur_up_step;
          cur_down_step = (255 - (u8)buf[data->loc_buf[index]]);
          if(cur_down_step > data->down_steps_buf[iter])
              data->down_steps_buf[iter] = cur_down_step;
      }
    }
    sum_steps = sum_steps + data->up_steps_buf[iter] + data->down_steps_buf[iter];
  }

  //reset counters
  data->iter_cnt = 0;
  data->ins_del_mut_cnt = data->num_of_ins_del_operations;

  /**
   * Return sum of all steps plus num_of_ins_del_operations of top locations
   **/
  return sum_steps + data->ins_del_mut_cnt;

}

/* Helper to clamp signed ints to a byte */
u8 clamp_to_byte(int32_t val) {
    if(val < 0)
          return 0;
      else if (val > 255)
          return 255;
      else
          return val;
}

// The following code is from the AFL++ sources (public domain)
// https://github.com/AFLplusplus/AFLplusplus/blob/stable/src/afl-performance.c
#define ROTL(d, lrot) ((d << (lrot)) | (d >> (8 * sizeof(d) - (lrot))))


// romuDuoJr
inline AFL_RAND_RETURN rand_next(afl_state_t *afl) {

  AFL_RAND_RETURN xp = afl->rand_seed[0];
  afl->rand_seed[0] = 15241094284759029579u * afl->rand_seed[1];
  afl->rand_seed[1] = afl->rand_seed[1] - xp;
  afl->rand_seed[1] = ROTL(afl->rand_seed[1], 27);
  return xp;

}

//The following functioni is from AFL++ (Apache-2.0)
//https://github.com/AFLplusplus/AFLplusplus/blob/7f17a94349830a54d2c899f56b149c0d7f9ffb9c/include/afl-mutations.h#L1751 
/* Helper to choose random block len for block operations in fuzz_one().
   Doesn't return zero, provided that max_len is > 0. */

static inline u32 choose_block_len(afl_state_t *afl, u32 limit) {

  u32 min_value, max_value;
  u32 rlim = MIN(afl->queue_cycle, (u32)3);

  if (unlikely(!afl->run_over10m)) { rlim = 1; }

  switch (rand_below(afl, rlim)) {

    case 0:
      min_value = 1;
      max_value = HAVOC_BLK_SMALL;
      break;

    case 1:
      min_value = HAVOC_BLK_SMALL;
      max_value = HAVOC_BLK_MEDIUM;
      break;

    default:

      if (likely(rand_below(afl, 10))) {

        min_value = HAVOC_BLK_MEDIUM;
        max_value = HAVOC_BLK_LARGE;

      } else {

        min_value = HAVOC_BLK_LARGE;
        max_value = HAVOC_BLK_XL;

      }

  }

  if (min_value >= limit) { min_value = 1; }

  return min_value + rand_below(afl, MIN(max_value, limit) - min_value + 1);

}

void check_valid_index(int index, int32_t* buf, size_t buf_size) {
  if(buf[index] >= buf_size){ //Should be ensured by python part
    WARNF("index points out of seed's length. Maybe disable trim!");
    buf[index] = 0; //Do something safe
  } 
}


/**
 * Perform custom mutations on a given input
 *
 * (Optional for now. Required in the future)
 *
 * @param[in] data pointer returned in afl_custom_init for this fuzz case
 * @param[in] buf Pointer to input data to be mutated
 * @param[in] buf_size Size of input data
 * @param[out] out_buf the buffer we will work on. we can reuse *buf. NULL on
 * error.
 * @param[in] add_buf Buffer containing the additional test case
 * @param[in] add_buf_size Size of the additional test case
 * @param[in] max_size Maximum size of the mutated output. The mutation must not
 *     produce data larger than max_size.
 * @return Size of the mutated output.
 */
size_t afl_custom_fuzz(ml_mutator_state_t *data, uint8_t *buf, size_t buf_size,
                       u8 **out_buf, uint8_t *add_buf,
                       size_t add_buf_size,  // add_buf can be NULL
                       size_t max_size) {

  if(data->afl->queue_cur->id < 0 ) {
    *out_buf = NULL;
    perror("Current ID < 0");
    return 0;            /* afl-fuzz will very likely error out after this. -> Fuzzing stops*/
  }

  /**
   * Here one test case is crafted
   **/

  *out_buf = buf;

  //check if previous iteration has been finished -> go to next
  while(data->up_steps_buf[data->iter_cnt] == 0 && data->down_steps_buf[data->iter_cnt] == 0 && data->iter_cnt < data->num_of_iterations) {
    data->iter_cnt++;
  }

  //get indices for current range of iteration (exponential increase)
  int low_index = data->iter_cnt == 0 ? 0 : pow(2, data->iter_cnt); // [14] = {0,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192};
  int up_index = pow(2, data->iter_cnt  + 1);

  if(data->up_steps_buf[data->iter_cnt] > 0) { //Up-Steps left?

    //do mutation for whole range of iteration
    for(int index=low_index; index<up_index; index++){
        //check for valid index
        check_valid_index(index, data->loc_buf, buf_size);
        //do gradient (or rather sign of gradient) guided mutation
        int  mut_val = buf[data->loc_buf[index]] + data->up_steps_buf[data->iter_cnt] * data->sign_buf[index];
        //clamp val to fit to byte
        buf[data->loc_buf[index]] = clamp_to_byte(mut_val);
    }
    data->up_steps_buf[data->iter_cnt]--;
    return buf_size; //Mutation done

  } else if (data->down_steps_buf[data->iter_cnt] > 0) { //Down-steps left?

    //do mutation for whole range of iteration
    for(int index=low_index; index<up_index; index++){
        //check for valid index
        check_valid_index(index, data->loc_buf, buf_size);
        //do gradient (or rather sign of gradient) guided mutation
        int  mut_val = buf[data->loc_buf[index]] - data->down_steps_buf[data->iter_cnt] * data->sign_buf[index];
        //clamp val to fit to byte
        buf[data->loc_buf[index]] = clamp_to_byte(mut_val);
    }
    data->down_steps_buf[data->iter_cnt]--;
    return buf_size; //Mutation done

  } else if(data->ins_del_mut_cnt > 0){

    /** Taken from NEUZZ...to be determined, how neccessary they are */
    //Now do deletions and insertions at top locations

    if(data->ins_del_mut_cnt % 2 == 0) { //alternating del ins operations
      int del_loc = data->loc_buf[data->ins_del_mut_cnt/2];
      //check for valid index
      check_valid_index(del_loc, data->loc_buf, buf_size);

      int cut_len = choose_block_len(data->afl, buf_size-del_loc);

      /* random deletion at a critical offset */
      //need memmove here because of overlapping ranges
      memmove(buf+del_loc, buf + del_loc+cut_len, buf_size - (del_loc+cut_len));


      data->ins_del_mut_cnt--;

      return buf_size - cut_len;

    } else { //ins operation
        int ins_loc = data->loc_buf[data->ins_del_mut_cnt/2];
        //check for valid index
        check_valid_index(ins_loc, data->loc_buf, buf_size);

        size_t cut_len = choose_block_len(data->afl, (buf_size-1) / 2); //divide by 2, so that cut_len + rand_loc < buf_size
        int rand_loc = (random()%cut_len);


        maybe_grow(BUF_PARAMS(data, ml_mutator), buf_size + cut_len);
        /* random insertion at a critical offset */
        memcpy(data->ml_mutator_buf, buf, ins_loc);
        memmove(data->ml_mutator_buf+ins_loc, data->ml_mutator_buf+rand_loc, cut_len);
        memmove(data->ml_mutator_buf+ins_loc+cut_len, data->ml_mutator_buf+ins_loc, buf_size - ins_loc );

        *out_buf = data->ml_mutator_buf;

        data->ins_del_mut_cnt--;

        return buf_size + cut_len;
    }


  } else { //Should not happen

    FATAL("All mutations have been applied, but mutator is requested for more?");
  }

  return buf_size; //should not be reached

}

  /**
   * Describe the current testcase, generated by the last mutation.
   * This will be called, for example, to give the written testcase a name
   * after a crash ocurred. It can help to reproduce crashing mutations.
   *
   * (Optional)
   *
   * @param data pointer returned by afl_customm_init for this custom mutator
   * @paramp[in] max_description_len maximum size avaliable for the description.
   *             A longer return string is legal, but will be truncated.
   * @return A valid ptr to a 0-terminated string.
   *         An empty or NULL return will result in a default description
   */
  const char* afl_custom_describe(ml_mutator_state_t* data, size_t max_description_len) {

    

    if(data->up_steps_buf[data->iter_cnt] != 0) {
      snprintf(data->describe_buffer, DESCRIBE_BUFFER_SIZE, "ml-mutator-up=%d,iter=%d", data->up_steps_buf[data->iter_cnt], data->iter_cnt);
    } else if(data->down_steps_buf[data->iter_cnt] != 0) {
      snprintf(data->describe_buffer, DESCRIBE_BUFFER_SIZE, "ml-mutator-down=%d,iter=%d", data->down_steps_buf[data->iter_cnt], data->iter_cnt);
    } else if(data->ins_del_mut_cnt % 2 == 0) { //del
      snprintf(data->describe_buffer, DESCRIBE_BUFFER_SIZE, "ml-mutator-del,iter=%d", data->iter_cnt);
    } else { //ins operation
      snprintf(data->describe_buffer, DESCRIBE_BUFFER_SIZE, "ml-mutator-ins,iter=%d", data->iter_cnt);
    }
    return data->describe_buffer;
  }



/**
 * Determine whether the fuzzer should fuzz the queue entry or not.
 *
 * (Optional)
 *
 * @param[in] data pointer returned in afl_custom_init for this fuzz case
 * @param filename File name of the test case in the queue entry
 * @return Return True(1) if the fuzzer will fuzz the queue entry, and
 *     False(0) otherwise.
 */
uint8_t afl_custom_queue_get(ml_mutator_state_t *data, const uint8_t *filename) {

  /**
   * Maybe skip entry if not yet incorparated in ML model
   * */
  return 1;
}


/**
 * Deinitialize everything
 *
 * @param data The data ptr from afl_custom_init
 */
void afl_custom_deinit(ml_mutator_state_t *data) {


  //Close named pipes
  fclose(data->out_pipe_fd);
  fclose(data->in_pipe_fd);

  if (data->ml_pid) {
    int status;
    waitpid(data->ml_pid, &status, 0); //wait till ml is done
    //We don't care about status since we just want it to end somehow
  }

  //Remove pipes
  unlink(data->out_pipe_file_name);
  unlink(data->in_pipe_file_name);

  free(data->ml_mutator_buf);
  free(data->up_steps_buf);
  free(data->down_steps_buf);
  free(data->out_pipe_file_name);
  free(data->in_pipe_file_name);

  if (data->gradient_file_line) {
    free(data->gradient_file_line);
  }
  

  free(data);
}
