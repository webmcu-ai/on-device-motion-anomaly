// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// MOTION ANOMALY DETECTION (2-CLASS ML + STATISTICAL FEATURES) — v005
//
// v005 DESIGN: back to real ML (per your feedback that you didn't want
// to lose the PSRAM-backed neural model). Built from the PROVEN 3-class
// Conv1D classifier (motion-original.ino) rather than the v001-v003
// autoencoder, because that autoencoder had 3 structural problems
// (documented in v004's header) that a from-scratch redesign couldn't
// patch around while staying single-class.
//
// The key change from the classifier: only TWO classes are collected -
//   0off    = device mounted on the machine, motor OFF
//   1normal = device mounted on the machine, motor running at normal speed
// This gives the network something to discriminate AGAINST during
// training (the same cross-entropy pressure that made 3-class motion
// classification work), instead of the old one-class reconstruction
// setup that had nothing to push against and collapsed to ~100% anomaly
// on real hardware.
//
// ANOMALY SCORE = 100% - P(1normal), i.e. how far the live softmax
// output has drifted from "this looks like normal running." As long as
// "normal" training data was captured at a normal, healthy running
// speed, drift toward "off-like" or generally away from that learned
// pattern (imbalance, bearing wear, mounting looseness, speed changes)
// should push this percentage up.
//
// PER-AXIS READOUT: a single Conv1D classifier mixes all 3 axes inside
// every filter, so it cannot cleanly report "which axis is anomalous."
// Per your choice, v005 keeps v004's per-axis RMS / peak-to-peak /
// zero-crossing-rate as a SEPARATE, secondary diagnostic readout,
// computed directly (no training needed) alongside the ML confidence
// score. Two independent signals: one trained/probabilistic (overall),
// one direct/statistical (per-axis detail).
//
// Input: 40 samples x 3 axes (AccelX, AccelY, AccelZ) = 120 floats
// Sampling: ~40 Hz over ~1 second per capture window
//
// SD card stores: sensor recordings in class folders (.csv)
// SD card stores: weights in binary format (.bin) and .h text header
// Serial monitor and OLED output
//
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique),
//   Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the U8g2 library declared in the platformio.ini file
// lib_deps = olikraus/U8g2 @ ^2.35.30
//            Seeed Arduino LSM6DS3
// board_build.arduino.memory_type = qio_opi
//


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Pins, Globals, Memory, Weights, Setup, Loop           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// Optional: uncomment AFTER copying myAnomalyMLWeights.h from SD to sketch folder
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS

#ifdef USE_BAKED_WEIGHTS
  #include "myAnomalyMLWeights.h"
#endif

#include <LSM6DS3.h>
#include <Wire.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>

// Create IMU object using I2C interface
LSM6DS3 myIMU(I2C_MODE, 0x6A);

U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// CONFIGURATION & ML HYPERPARAMETERS
// ======================================================

#define NUM_CLASSES 2

String myClassLabels[NUM_CLASSES] = {"0off", "1normal"};
const int myNormalClassIdx = 1;   // index of "1normal" - the class we score confidence against

const int myTotalItems = NUM_CLASSES + 2;  // classes + Train + Infer

float LEARNING_RATE  = 0.001f;
int   BATCH_SIZE     = 6;
int   TARGET_EPOCHS  = 30;
int   VALIDATION_SAMPLES = 3;   // last N samples per class held out for validation (0 = disabled)

// If ANY axis anomaly percent (from the secondary statistical readout)
// is >= this, OR the ML anomaly percent is >= this, flag ANOMALY overall.
float ANOMALY_FLAG_PERCENT = 50.0f;

// How many standard deviations away from the normal baseline counts as
// 100% anomalous for the secondary per-axis RMS/P2P/ZCR readout.
float FEATURE_SIGMA = 3.0f;

// ======================================================
// INPUT / NETWORK ARCHITECTURE CONSTANTS
// ======================================================

// Fixed input window: 40 time steps × 3 axes = 120 inputs
#define IMU_TIMESTEPS     40
#define IMU_AXES           3      // AccelX, AccelY, AccelZ
#define INPUT_SIZE       (IMU_TIMESTEPS * IMU_AXES)   // 120

// Sampling: ~40 Hz => one sample every 25 ms over ~1 second
#define SAMPLE_INTERVAL_MS  25

// Network: Conv1D → Dense → Output
//   Conv1D: kernel=5 over IMU_TIMESTEPS, IMU_AXES input channels, CONV1_FILTERS output channels
//   Applied as: for each output timestep t: sum over kernel positions and input axes
//   Output timesteps = IMU_TIMESTEPS - CONV1_KERNEL + 1 = 40 - 5 + 1 = 36
//   After max-pool /2: 18 timesteps × CONV1_FILTERS channels = CONV1_FLAT
//   Dense1: CONV1_FLAT → DENSE1_SIZE → NUM_CLASSES
#define CONV1_KERNEL    5
#define CONV1_FILTERS   8
#define CONV1_OUT_STEPS (IMU_TIMESTEPS - CONV1_KERNEL + 1)   // 36
#define POOL1_STEPS     (CONV1_OUT_STEPS / 2)                // 18
#define CONV1_FLAT      (POOL1_STEPS * CONV1_FILTERS)        // 18 × 8 = 144

// Conv1D weights: kernel × in_channels × out_filters
#define CONV1_WEIGHTS   (CONV1_KERNEL * IMU_AXES * CONV1_FILTERS)  // 5 × 3 × 8 = 120

#define DENSE1_SIZE   32
#define DENSE2_SIZE   16

#define DENSE1_WEIGHTS  (CONV1_FLAT  * DENSE1_SIZE)   // 144 × 32 = 4608
#define DENSE2_WEIGHTS  (DENSE1_SIZE * DENSE2_SIZE)   //  32 × 16 = 512
#define OUTPUT_WEIGHTS  (DENSE2_SIZE * NUM_CLASSES)   //  16 × NUM_CLASSES

// ======================================================
// SECONDARY PER-AXIS READOUT (from v004): RMS, peak-to-peak, and
// zero-crossing-rate per axis, computed directly - no training. This
// is what gives per-axis detail, since the Conv1D classifier above
// can only report one combined ML confidence number.
// ======================================================
#define FEATURES_PER_AXIS  3
#define TOTAL_FEATURES     (IMU_AXES * FEATURES_PER_AXIS)   // 9
enum { FEAT_RMS = 0, FEAT_P2P = 1, FEAT_ZCR = 2 };

float myFeatureMean[TOTAL_FEATURES] = {0};
float myFeatureStd [TOTAL_FEATURES] = {1,1,1, 1,1,1, 1,1,1};
int   myFeatureBaselineCount = 0;
bool  myFeatureBaselineReady = false;

// ======================================================
// NORMALIZATION CONSTANTS (computed at startup by myCalibrate())
// Defaults used only if calibration is skipped.
// Z axis default mean=1.0 accounts for gravity when device is flat.
// ======================================================
float myAccelMean[IMU_AXES] = { 0.0f,  0.0f,  1.0f };
float myAccelStd [IMU_AXES] = { 1.0f,  1.0f,  1.0f };
#define CALIB_SAMPLES  80   // ~2 seconds of stationary data at 40 Hz

// ======================================================
// TOUCH INPUT SYSTEM
// ======================================================
const int myThresholdPress   = 1100;
const int myThresholdRelease =  900;

struct TouchState {
  bool          isTouching     = false;
  int           tapCount       = 0;
  unsigned long firstTapTime   = 0;
  unsigned long lastReleaseTime= 0;
  unsigned long lastCheckTime  = 0;
  const unsigned long tapWindow    = 800;
  const int           longPressTaps= 3;
  const unsigned long debounceDelay= 50;
};

TouchState myTouch;

// ======================================================
// SYSTEM LOGIC VARIABLES
// ======================================================
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime      = 0;
const int     myTapCooldown      = 250;
int           myMenuIndex        = 1;
bool          myIsSelected       = false;
bool          myWeightsTrained   = false;
bool          mySDavailable      = false;

// ======================================================
// ML WEIGHT & GRADIENT BUFFERS (PSRAM)
// ======================================================
float* myInputBuffer = nullptr;   // INPUT_SIZE floats per inference/training step

// Conv1D weights and biases
float* myConv1_w = nullptr;       // CONV1_WEIGHTS = kernel × axes × filters
float* myConv1_b = nullptr;       // CONV1_FILTERS

// Dense weights and biases
float* myDense1_w = nullptr;
float* myDense1_b = nullptr;
float* myDense2_w = nullptr;
float* myDense2_b = nullptr;
float* myOutput_w = nullptr;
float* myOutput_b = nullptr;

// Gradients
float* myConv1_w_grad  = nullptr;
float* myConv1_b_grad  = nullptr;
float* myDense1_w_grad = nullptr;
float* myDense1_b_grad = nullptr;
float* myDense2_w_grad = nullptr;
float* myDense2_b_grad = nullptr;
float* myOutput_w_grad = nullptr;
float* myOutput_b_grad = nullptr;

// Adam optimizer momentum buffers
float* myConv1_w_m  = nullptr;  float* myConv1_w_v  = nullptr;
float* myConv1_b_m  = nullptr;  float* myConv1_b_v  = nullptr;
float* myDense1_w_m = nullptr;  float* myDense1_w_v = nullptr;
float* myDense1_b_m = nullptr;  float* myDense1_b_v = nullptr;
float* myDense2_w_m = nullptr;  float* myDense2_w_v = nullptr;
float* myDense2_b_m = nullptr;  float* myDense2_b_v = nullptr;
float* myOutput_w_m = nullptr;  float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr;  float* myOutput_b_v = nullptr;

// Forward-pass activation buffers
float* myConv1_output  = nullptr;   // CONV1_OUT_STEPS × CONV1_FILTERS (pre-pool)
float* myPool1_output  = nullptr;   // POOL1_STEPS     × CONV1_FILTERS = CONV1_FLAT (post-pool)
float* myDense1_output = nullptr;   // DENSE1_SIZE
float* myDense2_output = nullptr;   // DENSE2_SIZE
float* myFinal_output  = nullptr;   // NUM_CLASSES  (softmax probabilities)

// Backward-pass delta buffers
float* myOutput_delta = nullptr;   // NUM_CLASSES
float* myDense2_delta = nullptr;   // DENSE2_SIZE
float* myDense1_delta = nullptr;   // DENSE1_SIZE
float* myPool1_delta  = nullptr;   // CONV1_FLAT
float* myConv1_delta  = nullptr;   // CONV1_OUT_STEPS × CONV1_FILTERS

// Adam step counter
int myAdamStep = 0;
bool myCoreMemoryReady = false;

struct TrainingItem {
  String path;
  int    label;
};
std::vector<TrainingItem> myTrainingData;

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float myClip(float v, float mn=-100, float mx=100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}

inline float myLeakyRelu(float x)      { return x > 0 ? x : 0.1f * x; }
inline float myLeakyReluDeriv(float x) { return x > 0 ? 1.0f : 0.1f; }

// Softmax in-place over 'size' elements
void mySoftmax(float* x, int size) {
  float maxVal = x[0];
  for (int i = 1; i < size; i++) if (x[i] > maxVal) maxVal = x[i];
  float sum = 0;
  for (int i = 0; i < size; i++) { x[i] = exp(x[i] - maxVal); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

// Normalize one full input window using per-axis mean/std
// Input buffer layout: [t0_ax, t0_ay, t0_az, t1_ax, t1_ay, t1_az, ...]
void myNormalizeInput(float* buf) {
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    for (int a = 0; a < IMU_AXES; a++) {
      int idx = t * IMU_AXES + a;
      buf[idx] = (buf[idx] - myAccelMean[a]) / (myAccelStd[a] + 1e-8f);
      buf[idx] = myClip(buf[idx], -5.0f, 5.0f);
    }
  }
}

// ======================================================
// TOUCH INPUT FUNCTIONS
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching      = false;
  myTouch.tapCount        = 0;
  myTouch.firstTapTime    = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime   = 0;
}

void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;

  int  val         = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);

  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1;
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching      = false;
    myTouch.lastReleaseTime = now;
  }
}

// Returns: 0=no action, 1=tap, 2=long press (3+ taps)
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count  = myTouch.tapCount;
      myResetTouchState();
      Serial.printf(result == 2 ? "LONG PRESS (%d taps)\n" : "TAP (%d tap%s)\n",
                    count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}

// Non-blocking touch state update for use inside heavy computation loops
void myCheckTouchBackground() {
  myUpdateTouchState();
}

// ======================================================
// MEMORY ALLOCATION
// ======================================================
// Checks one allocation, prints a clear named error (with free-PSRAM
// context) instead of letting a null pointer slip through to a crash.
bool myCheckAlloc(void* ptr, const char* name, size_t bytes) {
  if (ptr) return true;
  Serial.printf("FATAL ALLOC FAIL: %s (%u bytes) - free PSRAM=%d, free heap=%d\n",
                name, (unsigned)bytes, ESP.getFreePsram(), ESP.getFreeHeap());
  return false;
}

// CORE allocation: weights + forward-pass buffers only. Everything
// Collect / Infer / boot ever needs. Stays allocated for program life.
bool myAllocateCoreMemory() {
  if (myCoreMemoryReady) return true;
  Serial.println("\n=== Allocating Core Memory (weights + forward buffers) ===");
  Serial.printf("Free PSRAM before allocation: %d bytes\n", ESP.getFreePsram());

  bool ok = true;
  #define MY_ALLOC_CORE(ptr, size, name) \
    ptr = (float*)ps_malloc((size) * sizeof(float)); \
    ok &= myCheckAlloc(ptr, name, (size) * sizeof(float));

  MY_ALLOC_CORE(myInputBuffer, INPUT_SIZE, "myInputBuffer");

  MY_ALLOC_CORE(myConv1_w, CONV1_WEIGHTS, "myConv1_w");
  MY_ALLOC_CORE(myConv1_b, CONV1_FILTERS, "myConv1_b");
  MY_ALLOC_CORE(myDense1_w, DENSE1_WEIGHTS, "myDense1_w");
  MY_ALLOC_CORE(myDense1_b, DENSE1_SIZE, "myDense1_b");
  MY_ALLOC_CORE(myDense2_w, DENSE2_WEIGHTS, "myDense2_w");
  MY_ALLOC_CORE(myDense2_b, DENSE2_SIZE, "myDense2_b");
  MY_ALLOC_CORE(myOutput_w, OUTPUT_WEIGHTS, "myOutput_w");
  MY_ALLOC_CORE(myOutput_b, NUM_CLASSES, "myOutput_b");

  MY_ALLOC_CORE(myConv1_output, CONV1_OUT_STEPS * CONV1_FILTERS, "myConv1_output");
  MY_ALLOC_CORE(myPool1_output, CONV1_FLAT, "myPool1_output");
  MY_ALLOC_CORE(myDense1_output, DENSE1_SIZE, "myDense1_output");
  MY_ALLOC_CORE(myDense2_output, DENSE2_SIZE, "myDense2_output");
  MY_ALLOC_CORE(myFinal_output, NUM_CLASSES, "myFinal_output");

  #undef MY_ALLOC_CORE

  if (!ok) return false;
  Serial.printf("Free PSRAM after core allocation: %d bytes\n", ESP.getFreePsram());

  // He initialization
  float c1std = sqrt(2.0f / (CONV1_KERNEL * IMU_AXES));
  for (int i = 0; i < CONV1_WEIGHTS; i++)
    myConv1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c1std;
  for (int i = 0; i < CONV1_FILTERS; i++) myConv1_b[i] = 0;

  float d1std = sqrt(2.0f / CONV1_FLAT);
  for (int i = 0; i < DENSE1_WEIGHTS; i++)
    myDense1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * d1std;
  for (int i = 0; i < DENSE1_SIZE; i++) myDense1_b[i] = 0;

  float d2std = sqrt(2.0f / DENSE1_SIZE);
  for (int i = 0; i < DENSE2_WEIGHTS; i++)
    myDense2_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * d2std;
  for (int i = 0; i < DENSE2_SIZE; i++) myDense2_b[i] = 0;

  float ostd = sqrt(2.0f / DENSE2_SIZE);
  for (int i = 0; i < OUTPUT_WEIGHTS; i++)
    myOutput_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * ostd;
  for (int i = 0; i < NUM_CLASSES; i++) myOutput_b[i] = 0;

  Serial.println("He-init random weights set");
  myCoreMemoryReady = true;
  return true;
}

// TRAINING-ONLY allocation: gradients + Adam m/v. Allocated at the
// start of Train, freed right after - keeps steady-state PSRAM use
// down to ~1/4 of what it would be if these stayed resident.
bool myAllocateTrainingMemory() {
  Serial.println("\n=== Allocating Training-Only Memory (grad + Adam) ===");
  Serial.printf("Free PSRAM before allocation: %d bytes\n", ESP.getFreePsram());

  bool ok = true;
  #define MY_ALLOC_GRAD(ptr, size, name) \
    ptr = (float*)ps_malloc((size) * sizeof(float)); \
    ok &= myCheckAlloc(ptr, name, (size) * sizeof(float));
  #define MY_ALLOC_ADAM(ptr, size, name) \
    ptr = (float*)ps_calloc((size), sizeof(float)); \
    ok &= myCheckAlloc(ptr, name, (size) * sizeof(float));

  MY_ALLOC_GRAD(myConv1_w_grad,  CONV1_WEIGHTS,  "myConv1_w_grad");
  MY_ALLOC_GRAD(myConv1_b_grad,  CONV1_FILTERS,  "myConv1_b_grad");
  MY_ALLOC_GRAD(myDense1_w_grad, DENSE1_WEIGHTS, "myDense1_w_grad");
  MY_ALLOC_GRAD(myDense1_b_grad, DENSE1_SIZE,    "myDense1_b_grad");
  MY_ALLOC_GRAD(myDense2_w_grad, DENSE2_WEIGHTS, "myDense2_w_grad");
  MY_ALLOC_GRAD(myDense2_b_grad, DENSE2_SIZE,    "myDense2_b_grad");
  MY_ALLOC_GRAD(myOutput_w_grad, OUTPUT_WEIGHTS, "myOutput_w_grad");
  MY_ALLOC_GRAD(myOutput_b_grad, NUM_CLASSES,    "myOutput_b_grad");

  MY_ALLOC_ADAM(myConv1_w_m, CONV1_WEIGHTS, "myConv1_w_m");  MY_ALLOC_ADAM(myConv1_w_v, CONV1_WEIGHTS, "myConv1_w_v");
  MY_ALLOC_ADAM(myConv1_b_m, CONV1_FILTERS, "myConv1_b_m");  MY_ALLOC_ADAM(myConv1_b_v, CONV1_FILTERS, "myConv1_b_v");
  MY_ALLOC_ADAM(myDense1_w_m, DENSE1_WEIGHTS, "myDense1_w_m");  MY_ALLOC_ADAM(myDense1_w_v, DENSE1_WEIGHTS, "myDense1_w_v");
  MY_ALLOC_ADAM(myDense1_b_m, DENSE1_SIZE,    "myDense1_b_m");  MY_ALLOC_ADAM(myDense1_b_v, DENSE1_SIZE,    "myDense1_b_v");
  MY_ALLOC_ADAM(myDense2_w_m, DENSE2_WEIGHTS, "myDense2_w_m");  MY_ALLOC_ADAM(myDense2_w_v, DENSE2_WEIGHTS, "myDense2_w_v");
  MY_ALLOC_ADAM(myDense2_b_m, DENSE2_SIZE,    "myDense2_b_m");  MY_ALLOC_ADAM(myDense2_b_v, DENSE2_SIZE,    "myDense2_b_v");
  MY_ALLOC_ADAM(myOutput_w_m, OUTPUT_WEIGHTS, "myOutput_w_m");  MY_ALLOC_ADAM(myOutput_w_v, OUTPUT_WEIGHTS, "myOutput_w_v");
  MY_ALLOC_ADAM(myOutput_b_m, NUM_CLASSES,    "myOutput_b_m");  MY_ALLOC_ADAM(myOutput_b_v, NUM_CLASSES,    "myOutput_b_v");

  MY_ALLOC_GRAD(myOutput_delta, NUM_CLASSES,                     "myOutput_delta");
  MY_ALLOC_GRAD(myDense2_delta, DENSE2_SIZE,                     "myDense2_delta");
  MY_ALLOC_GRAD(myDense1_delta, DENSE1_SIZE,                     "myDense1_delta");
  MY_ALLOC_GRAD(myPool1_delta,  CONV1_FLAT,                      "myPool1_delta");
  MY_ALLOC_GRAD(myConv1_delta,  CONV1_OUT_STEPS * CONV1_FILTERS, "myConv1_delta");

  #undef MY_ALLOC_GRAD
  #undef MY_ALLOC_ADAM

  Serial.printf("Free PSRAM after training allocation: %d bytes\n", ESP.getFreePsram());
  myAdamStep = 0;   // fresh Adam state for this training session
  return ok;
}

// Frees every training-only buffer and resets pointers to nullptr.
void myFreeTrainingMemory() {
  float** myTrainingPtrs[] = {
    &myConv1_w_grad, &myConv1_b_grad, &myDense1_w_grad, &myDense1_b_grad,
    &myDense2_w_grad, &myDense2_b_grad, &myOutput_w_grad, &myOutput_b_grad,
    &myConv1_w_m, &myConv1_w_v, &myConv1_b_m, &myConv1_b_v,
    &myDense1_w_m, &myDense1_w_v, &myDense1_b_m, &myDense1_b_v,
    &myDense2_w_m, &myDense2_w_v, &myDense2_b_m, &myDense2_b_v,
    &myOutput_w_m, &myOutput_w_v, &myOutput_b_m, &myOutput_b_v,
    &myOutput_delta, &myDense2_delta, &myDense1_delta, &myPool1_delta, &myConv1_delta
  };
  for (float** p : myTrainingPtrs) {
    if (*p) { free(*p); *p = nullptr; }
  }
  Serial.printf("Training memory freed. Free PSRAM now: %d bytes\n", ESP.getFreePsram());
}

// ======================================================
// WEIGHT SAVE / LOAD / EXPORT
// ======================================================
void myExportHeader() {
  if (!mySDavailable) { Serial.println("No SD card - cannot export header"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myAnomalyMLWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MOTION_MODEL_H\n#define MY_MOTION_MODEL_H");
  file.println("// Uncomment in main sketch:  #define USE_BAKED_WEIGHTS");
  file.printf( "// #define NUM_CLASSES %d\n", NUM_CLASSES);
  file.print("// String myClassLabels[NUM_CLASSES] = {");
  for (int i = 0; i < NUM_CLASSES; i++) {
    file.printf("\"%s\"", myClassLabels[i].c_str());
    if (i < NUM_CLASSES - 1) file.print(", ");
  }
  file.println("};");

  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for (int i = 0; i < size; i++) {
      file.print(data[i], 6); file.print("f");
      if (i < size - 1) file.print(", ");
      if ((i + 1) % 8 == 0) file.println();
    }
    file.println(" };");
  };
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_dense1_w", myDense1_w, DENSE1_WEIGHTS);
  myDump("myModel_dense1_b", myDense1_b, DENSE1_SIZE);
  myDump("myModel_dense2_w", myDense2_w, DENSE2_WEIGHTS);
  myDump("myModel_dense2_b", myDense2_b, DENSE2_SIZE);
  myDump("myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDump("myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#endif");
  file.close();
  Serial.println("Header exported to /header/myAnomalyMLWeights.h");
}

bool myLoadWeights() {
  if (!mySDavailable) { Serial.println("No SD - skipping weight load"); return false; }
  if (!SD.exists("/header/myAnomalyMLWeights.bin")) { Serial.println("No weights file found"); return false; }
  Serial.println("Loading weights from SD...");
  File f = SD.open("/header/myAnomalyMLWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
  f.read((uint8_t*)myDense1_w, DENSE1_WEIGHTS * 4);
  f.read((uint8_t*)myDense1_b, DENSE1_SIZE    * 4);
  f.read((uint8_t*)myDense2_w, DENSE2_WEIGHTS * 4);
  f.read((uint8_t*)myDense2_b, DENSE2_SIZE    * 4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
  f.close();
  Serial.println("Weights loaded successfully");
  myWeightsTrained = true;
  return true;
}

void mySaveWeights() {
  if (!mySDavailable) { Serial.println("No SD - cannot save weights"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myAnomalyMLWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
    f.write((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
    f.write((uint8_t*)myDense1_w, DENSE1_WEIGHTS * 4);
    f.write((uint8_t*)myDense1_b, DENSE1_SIZE    * 4);
    f.write((uint8_t*)myDense2_w, DENSE2_WEIGHTS * 4);
    f.write((uint8_t*)myDense2_b, DENSE2_SIZE    * 4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
    f.write((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
    f.close();
    Serial.println("Weights saved to SD");
  }
  myExportHeader();
}

// ======================================================
// CALIBRATION  (run once at startup, device stationary)
// Collects CALIB_SAMPLES readings and computes per-axis mean and std.
// Saves result to SD so subsequent boots skip the wait.
// ======================================================
void myCalibrate() {
  // Try loading from SD first
  if (mySDavailable && SD.exists("/header/myCalib.bin")) {
    File f = SD.open("/header/myCalib.bin", FILE_READ);
    if (f && f.size() == IMU_AXES * 2 * 4) {
      f.read((uint8_t*)myAccelMean, IMU_AXES * 4);
      f.read((uint8_t*)myAccelStd,  IMU_AXES * 4);
      f.close();
      Serial.printf("Calibration loaded: mean=%.3f,%.3f,%.3f  std=%.3f,%.3f,%.3f\n",
                    myAccelMean[0], myAccelMean[1], myAccelMean[2],
                    myAccelStd[0],  myAccelStd[1],  myAccelStd[2]);
      return;
    }
    if (f) f.close();
  }

  Serial.println("Calibrating IMU - keep device stationary...");
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 10, "Calibrating...");
    u8g2.drawStr(0, 22, "Keep still!");
  } while (u8g2.nextPage());
  delay(500);

  float sum[IMU_AXES]  = {0, 0, 0};
  float sum2[IMU_AXES] = {0, 0, 0};

  for (int i = 0; i < CALIB_SAMPLES; i++) {
    float v[IMU_AXES];
    v[0] = myIMU.readFloatAccelX();
    v[1] = myIMU.readFloatAccelY();
    v[2] = myIMU.readFloatAccelZ();
    for (int a = 0; a < IMU_AXES; a++) { sum[a] += v[a]; sum2[a] += v[a] * v[a]; }
    delay(SAMPLE_INTERVAL_MS);
  }

  for (int a = 0; a < IMU_AXES; a++) {
    myAccelMean[a] = sum[a] / CALIB_SAMPLES;
    float var = (sum2[a] / CALIB_SAMPLES) - (myAccelMean[a] * myAccelMean[a]);
    myAccelStd[a]  = max(sqrt(var), 0.01f);   // floor at 0.01 to avoid div/0
  }

  Serial.printf("Calibration done: mean=%.3f,%.3f,%.3f  std=%.3f,%.3f,%.3f\n",
                myAccelMean[0], myAccelMean[1], myAccelMean[2],
                myAccelStd[0],  myAccelStd[1],  myAccelStd[2]);

  // Save to SD for next boot
  if (mySDavailable) {
    if (!SD.exists("/header")) SD.mkdir("/header");
    File f = SD.open("/header/myCalib.bin", FILE_WRITE);
    if (f) {
      f.write((uint8_t*)myAccelMean, IMU_AXES * 4);
      f.write((uint8_t*)myAccelStd,  IMU_AXES * 4);
      f.close();
      Serial.println("Calibration saved to SD");
    }
  }
}

// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void myActionCollect(int classIdx);
void myActionTrain();
void myActionInfer();
void myResetMenuState();
void myHandleMenuNavigation();
void myDrawMenu();

// ======================================================
// SETUP AND LOOP
// ======================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32-S3 Motion ML System Starting ===");
  Serial.printf("Free heap:  %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  pinMode(A0, INPUT);
  u8g2.begin();

  // SD card init
  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);
  Serial.println("Checking SD card...");
  SPI.begin();
  SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("=================================================");
    Serial.println("WARNING: No SD card detected!");
    Serial.println("  Collect / Train / weight-save all require an SD");
    Serial.println("  card. Infer will only work if baked-in weights");
    Serial.println("  (USE_BAKED_WEIGHTS) and/or a baked feature");
    Serial.println("  baseline are compiled in.");
    Serial.println("=================================================");
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.drawStr(0, 10, "NO SD CARD!");
      u8g2.drawStr(0, 22, "Insert card and");
      u8g2.drawStr(0, 32, "reset device");
    } while (u8g2.nextPage());
    delay(3500);
  } else {
    Serial.println("SD card mounted successfully");
  }

  // IMU init
  if (myIMU.begin() != 0) {
    Serial.println("ERROR: IMU initialization failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "IMU ERROR!"); } while (u8g2.nextPage());
    while (1) { delay(1000); }
  }
  Serial.println("IMU initialized successfully");
  myCalibrate();

  if (!myAllocateCoreMemory()) {
    Serial.println("FATAL: Core PSRAM allocation failed - see allocation errors above.");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 12, "PSRAM ERROR!"); u8g2.drawStr(0, 26, "See Serial log"); } while (u8g2.nextPage());
    while (1) { delay(1000); }
  }

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myDense1_w, myModel_dense1_w, DENSE1_WEIGHTS * sizeof(float));
  memcpy(myDense1_b, myModel_dense1_b, DENSE1_SIZE    * sizeof(float));
  memcpy(myDense2_w, myModel_dense2_w, DENSE2_WEIGHTS * sizeof(float));
  memcpy(myDense2_b, myModel_dense2_b, DENSE2_SIZE    * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  Serial.println("Baked-in weights loaded");
  myWeightsTrained = true;
#endif

  if (myLoadWeights()) {
    Serial.println("SD weights loaded - overriding baked-in weights");
  }

  myLoadFeatureBaseline();   // secondary per-axis RMS/P2P/ZCR baseline (optional, not fatal if missing)

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);
  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
}



// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: DATA COLLECTION FUNCTIONS                                       ██
// ██                                                                          ██
// ██  Captures one 1-second IMU window and saves it to SD as a .csv file.     ██
// ██  File format: one row per timestep, columns: ax,ay,az                    ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// Count .csv files in a class folder
int myCountSamples(int classIdx) {
  if (!mySDavailable) return 0;
  String path = "/motion/" + myClassLabels[classIdx];
  File root = SD.open(path);
  if (!root) return 0;
  int count = 0;
  while (File f = root.openNextFile()) {
    if (!f.isDirectory() && String(f.name()).endsWith(".csv")) count++;
    f.close();
  }
  root.close();
  return count;
}

// Capture one IMU window: 40 samples at ~25 ms intervals, save to SD
bool myCaptureSample(int classIdx) {
  String folderPath = "/motion/" + myClassLabels[classIdx];
  if (!SD.exists("/motion")) SD.mkdir("/motion");
  if (!SD.exists(folderPath)) SD.mkdir(folderPath);

  int sampleNum = myCountSamples(classIdx);
  String filePath = folderPath + "/s" + String(sampleNum) + ".csv";

  File f = SD.open(filePath, FILE_WRITE);
  if (!f) { Serial.println("ERROR: cannot open file for writing"); return false; }

  Serial.printf("Capturing %d samples to %s\n", IMU_TIMESTEPS, filePath.c_str());

  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    unsigned long tStart = millis();
    float ax = myIMU.readFloatAccelX();
    float ay = myIMU.readFloatAccelY();
    float az = myIMU.readFloatAccelZ();

    f.printf("%.5f,%.5f,%.5f\n", ax, ay, az);

    // Brief echo every 10 samples
    if (t % 10 == 0) Serial.printf("  t%02d: %.3f,%.3f,%.3f\n", t, ax, ay, az);

    // Pace to SAMPLE_INTERVAL_MS
    long elapsed = millis() - tStart;
    if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
  }
  f.close();
  Serial.printf("Saved: %s\n", filePath.c_str());
  return true;
}

void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect samples");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("TAP (1-2 taps) = Capture 1-second window");
  Serial.println("LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("Serial: 't'=capture, 'l'=exit");

  myResetTouchState();
  int captureCount = myCountSamples(classIdx);

  // Show OLED prompt
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 8,  myClassLabels[classIdx].c_str());
    u8g2.drawStr(0, 18, "TAP=Capture");
    u8g2.drawStr(0, 28, "HOLD=Exit");
    char buf[20]; snprintf(buf, sizeof(buf), "Count: %d", captureCount);
    u8g2.drawStr(0, 38, buf);
  } while (u8g2.nextPage());

  while (true) {
    // Serial input
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'l' || c == 'L') { myResetMenuState(); return; }
      if (c == 't' || c == 'T') {
        Serial.println("Hold still... capturing in 1s");
        delay(1000);
        if (myCaptureSample(classIdx)) {
          captureCount++;
          Serial.printf("Total samples for %s: %d\n", myClassLabels[classIdx].c_str(), captureCount);
          u8g2.firstPage();
          do {
            u8g2.setFont(u8g2_font_5x7_tf);
            u8g2.drawStr(0, 8,  myClassLabels[classIdx].c_str());
            char buf[20]; snprintf(buf, sizeof(buf), "Saved: %d", captureCount);
            u8g2.drawStr(0, 20, buf);
            u8g2.drawStr(0, 32, "TAP=More");
          } while (u8g2.nextPage());
        }
      }
    }

    // Touch input
    int touchAction = myCheckTouchInput();
    if (touchAction == 2) { myResetMenuState(); return; }
    if (touchAction == 1) {
      Serial.println("Hold still... capturing in 1s");
      delay(1000);
      if (myCaptureSample(classIdx)) {
        captureCount++;
        Serial.printf("Total samples for %s: %d\n", myClassLabels[classIdx].c_str(), captureCount);
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.drawStr(0, 8,  myClassLabels[classIdx].c_str());
          char buf[20]; snprintf(buf, sizeof(buf), "Saved: %d", captureCount);
          u8g2.drawStr(0, 20, buf);
          u8g2.drawStr(0, 32, "TAP=More");
        } while (u8g2.nextPage());
      }
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: FORWARD PASS, BACKWARD PASS, TRAINING                          ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// Conv1D forward pass
// Input layout: [t0_ax, t0_ay, t0_az, t1_ax, ...]  (IMU_TIMESTEPS × IMU_AXES)
// Weight layout: [k × in_axis × out_filter]  index = (k*IMU_AXES + a)*CONV1_FILTERS + f
// Output layout: [step × filter]  index = step*CONV1_FILTERS + f
void myConv1DForward(float* input) {
  for (int s = 0; s < CONV1_OUT_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float sum = myConv1_b[f];
      for (int k = 0; k < CONV1_KERNEL; k++) {
        for (int a = 0; a < IMU_AXES; a++) {
          sum += input[(s + k) * IMU_AXES + a] * myConv1_w[(k * IMU_AXES + a) * CONV1_FILTERS + f];
        }
      }
      myConv1_output[s * CONV1_FILTERS + f] = myLeakyRelu(sum);
    }
  }
}

// Max-pool /2 along the time axis, per filter
void myPool1Forward() {
  for (int s = 0; s < POOL1_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float a = myConv1_output[(s * 2)     * CONV1_FILTERS + f];
      float b = myConv1_output[(s * 2 + 1) * CONV1_FILTERS + f];
      myPool1_output[s * CONV1_FILTERS + f] = max(a, b);
    }
  }
}

// Dense layer forward: output[j] = leaky_relu( sum_i(w[i*outSize+j] * input[i]) + b[j] )
void myDenseForward(float* input, int inSize,
                    float* w, float* b,
                    float* output, int outSize,
                    bool applyActivation) {
  for (int j = 0; j < outSize; j++) {
    float sum = b[j];
    for (int i = 0; i < inSize; i++) sum += input[i] * w[i * outSize + j];
    output[j] = applyActivation ? myLeakyRelu(sum) : sum;
  }
}

// Full forward pass: Conv1D → Pool → Dense1 → Dense2 → Output(softmax)
void myForwardPass(float* input) {
  myConv1DForward(input);
  myPool1Forward();
  myDenseForward(myPool1_output, CONV1_FLAT,  myDense1_w, myDense1_b, myDense1_output, DENSE1_SIZE, true);
  myDenseForward(myDense1_output, DENSE1_SIZE, myDense2_w, myDense2_b, myDense2_output, DENSE2_SIZE, true);
  myDenseForward(myDense2_output, DENSE2_SIZE, myOutput_w, myOutput_b, myFinal_output,  NUM_CLASSES, false);
  mySoftmax(myFinal_output, NUM_CLASSES);
}

// Cross-entropy loss for one sample (label is integer class index)
float myComputeLoss(int label) {
  float p = max(myFinal_output[label], 1e-7f);
  return -log(p);
}

// Adam update helper for one parameter array
void myAdamUpdate(float* w, float* grad, float* m, float* v, int size, float lr) {
  const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
  myAdamStep++;
  float bc1 = 1.0f - pow(beta1, myAdamStep);
  float bc2 = 1.0f - pow(beta2, myAdamStep);
  for (int i = 0; i < size; i++) {
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
    float mHat = m[i] / bc1;
    float vHat = v[i] / bc2;
    w[i] -= lr * mHat / (sqrt(vHat) + eps);
  }
}

// Zero all gradient buffers
void myZeroGradients() {
  memset(myConv1_w_grad,  0, CONV1_WEIGHTS  * sizeof(float));
  memset(myConv1_b_grad,  0, CONV1_FILTERS  * sizeof(float));
  memset(myDense1_w_grad, 0, DENSE1_WEIGHTS * sizeof(float));
  memset(myDense1_b_grad, 0, DENSE1_SIZE    * sizeof(float));
  memset(myDense2_w_grad, 0, DENSE2_WEIGHTS * sizeof(float));
  memset(myDense2_b_grad, 0, DENSE2_SIZE    * sizeof(float));
  memset(myOutput_w_grad, 0, OUTPUT_WEIGHTS * sizeof(float));
  memset(myOutput_b_grad, 0, NUM_CLASSES    * sizeof(float));
}

// Backward pass for one sample, accumulates gradients into all layers
void myBackwardPass(float* input, int label) {
  // --- Output layer: softmax + cross-entropy combined ---
  for (int j = 0; j < NUM_CLASSES; j++)
    myOutput_delta[j] = myFinal_output[j] - (j == label ? 1.0f : 0.0f);

  for (int i = 0; i < DENSE2_SIZE; i++)
    for (int j = 0; j < NUM_CLASSES; j++)
      myOutput_w_grad[i * NUM_CLASSES + j] += myDense2_output[i] * myOutput_delta[j];
  for (int j = 0; j < NUM_CLASSES; j++) myOutput_b_grad[j] += myOutput_delta[j];

  // --- Dense2 ---
  for (int i = 0; i < DENSE2_SIZE; i++) {
    float sum = 0;
    for (int j = 0; j < NUM_CLASSES; j++) sum += myOutput_w[i * NUM_CLASSES + j] * myOutput_delta[j];
    myDense2_delta[i] = sum * myLeakyReluDeriv(myDense2_output[i]);
  }
  for (int i = 0; i < DENSE1_SIZE; i++)
    for (int j = 0; j < DENSE2_SIZE; j++)
      myDense2_w_grad[i * DENSE2_SIZE + j] += myDense1_output[i] * myDense2_delta[j];
  for (int j = 0; j < DENSE2_SIZE; j++) myDense2_b_grad[j] += myDense2_delta[j];

  // --- Dense1 ---
  for (int i = 0; i < DENSE1_SIZE; i++) {
    float sum = 0;
    for (int j = 0; j < DENSE2_SIZE; j++) sum += myDense2_w[i * DENSE2_SIZE + j] * myDense2_delta[j];
    myDense1_delta[i] = sum * myLeakyReluDeriv(myDense1_output[i]);
  }
  for (int i = 0; i < CONV1_FLAT; i++)
    for (int j = 0; j < DENSE1_SIZE; j++)
      myDense1_w_grad[i * DENSE1_SIZE + j] += myPool1_output[i] * myDense1_delta[j];
  for (int j = 0; j < DENSE1_SIZE; j++) myDense1_b_grad[j] += myDense1_delta[j];

  // --- Max-pool backward: route gradient to whichever input was the max ---
  for (int s = 0; s < POOL1_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float grad = 0;
      for (int j = 0; j < DENSE1_SIZE; j++)
        grad += myDense1_w[((s * CONV1_FILTERS + f)) * DENSE1_SIZE + j] * myDense1_delta[j];
      myPool1_delta[s * CONV1_FILTERS + f] = grad;

      float a = myConv1_output[(s * 2)     * CONV1_FILTERS + f];
      float b = myConv1_output[(s * 2 + 1) * CONV1_FILTERS + f];
      myConv1_delta[(s * 2)     * CONV1_FILTERS + f] = (a >= b) ? grad : 0.0f;
      myConv1_delta[(s * 2 + 1) * CONV1_FILTERS + f] = (b >  a) ? grad : 0.0f;
    }
  }

  // --- Conv1D backward ---
  for (int s = 0; s < CONV1_OUT_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float delta = myConv1_delta[s * CONV1_FILTERS + f]
                    * myLeakyReluDeriv(myConv1_output[s * CONV1_FILTERS + f]);
      myConv1_b_grad[f] += delta;
      for (int k = 0; k < CONV1_KERNEL; k++) {
        for (int a = 0; a < IMU_AXES; a++) {
          myConv1_w_grad[(k * IMU_AXES + a) * CONV1_FILTERS + f] +=
            input[(s + k) * IMU_AXES + a] * delta;
        }
      }
    }
  }
}

// Load one .csv sample from SD into buf (INPUT_SIZE floats) then normalize
bool myLoadSampleFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if (!f) return false;
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    for (int a = 0; a < IMU_AXES; a++) {
      buf[t * IMU_AXES + a] = f.parseFloat();
      if (a < IMU_AXES - 1) {
        // consume the comma
        while (f.available() && f.peek() == ',') f.read();
      }
    }
    // consume newline
    while (f.available() && (f.peek() == '\n' || f.peek() == '\r')) f.read();
  }
  f.close();
  myNormalizeInput(buf);
  return true;
}

// ======================================================
// SECONDARY PER-AXIS READOUT (from v004): RMS / peak-to-peak /
// zero-crossing-rate per axis, computed directly - no training needed.
// This is what gives per-axis granularity, since the Conv1D classifier
// above only produces one combined confidence number.
// ======================================================

// Removes stationary gravity/offset per axis WITHOUT dividing by std,
// so vibration AMPLITUDE (which the features below measure) survives.
void myMeanCenterOnly(float* buf) {
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    for (int a = 0; a < IMU_AXES; a++) {
      int idx = t * IMU_AXES + a;
      buf[idx] = myClip(buf[idx] - myAccelMean[a], -20.0f, 20.0f);
    }
  }
}

void myExtractFeatures(float* window, float featOut[TOTAL_FEATURES]) {
  for (int a = 0; a < IMU_AXES; a++) {
    float sumSq = 0;
    float minV = 1e9f, maxV = -1e9f;
    int   zeroCrossings = 0;
    float prevSign = 0;

    for (int t = 0; t < IMU_TIMESTEPS; t++) {
      float v = window[t * IMU_AXES + a];
      sumSq += v * v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
      float sign = (v >= 0) ? 1.0f : -1.0f;
      if (t > 0 && sign != prevSign) zeroCrossings++;
      prevSign = sign;
    }

    featOut[a * FEATURES_PER_AXIS + FEAT_RMS] = sqrt(sumSq / IMU_TIMESTEPS);
    featOut[a * FEATURES_PER_AXIS + FEAT_P2P] = maxV - minV;
    featOut[a * FEATURES_PER_AXIS + FEAT_ZCR] = (float)zeroCrossings / (IMU_TIMESTEPS - 1);
  }
}

bool myLoadAndExtractFeatures(const char* path, float featOut[TOTAL_FEATURES]) {
  float buf[INPUT_SIZE];   // stack only - no PSRAM needed for this readout
  File f = SD.open(path);
  if (!f) return false;
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    for (int a = 0; a < IMU_AXES; a++) {
      buf[t * IMU_AXES + a] = f.parseFloat();
      if (a < IMU_AXES - 1) while (f.available() && f.peek() == ',') f.read();
    }
    while (f.available() && (f.peek() == '\n' || f.peek() == '\r')) f.read();
  }
  f.close();
  myMeanCenterOnly(buf);
  myExtractFeatures(buf, featOut);
  return true;
}

bool myLoadFeatureBaseline() {
  if (!mySDavailable) return false;
  if (!SD.exists("/header/myFeatureBaseline.bin")) return false;
  File f = SD.open("/header/myFeatureBaseline.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myFeatureMean, TOTAL_FEATURES * 4);
  f.read((uint8_t*)myFeatureStd,  TOTAL_FEATURES * 4);
  f.read((uint8_t*)&myFeatureBaselineCount, sizeof(int));
  f.close();
  Serial.printf("Feature baseline loaded (n=%d)\n", myFeatureBaselineCount);
  myFeatureBaselineReady = (myFeatureBaselineCount > 0);
  return true;
}

void mySaveFeatureBaseline() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myFeatureBaseline.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myFeatureMean, TOTAL_FEATURES * 4);
    f.write((uint8_t*)myFeatureStd,  TOTAL_FEATURES * 4);
    f.write((uint8_t*)&myFeatureBaselineCount, sizeof(int));
    f.close();
    Serial.println("Feature baseline saved to SD");
  }
}

// Computes the RMS/P2P/ZCR baseline from ONLY the "1normal" class
// captures (the "0off" captures aren't representative of running
// vibration and shouldn't be part of this baseline).
void myComputeFeatureBaseline() {
  String path = "/motion/" + myClassLabels[myNormalClassIdx];
  File root = SD.open(path);
  if (!root) { Serial.println("No normal-class folder found for feature baseline."); return; }

  float sum[TOTAL_FEATURES]  = {0};
  float sum2[TOTAL_FEATURES] = {0};
  int n = 0;

  while (File file = root.openNextFile()) {
    String name = file.name();
    bool isCsv = !file.isDirectory() && name.endsWith(".csv");
    file.close();
    if (!isCsv) continue;
    myCheckTouchBackground();
    float feat[TOTAL_FEATURES];
    if (!myLoadAndExtractFeatures((path + "/" + name).c_str(), feat)) continue;
    for (int k = 0; k < TOTAL_FEATURES; k++) { sum[k] += feat[k]; sum2[k] += feat[k] * feat[k]; }
    n++;
  }
  root.close();

  if (n == 0) { Serial.println("No normal-class samples - feature baseline not updated."); return; }

  for (int k = 0; k < TOTAL_FEATURES; k++) {
    myFeatureMean[k] = sum[k] / n;
    float var = (sum2[k] / n) - (myFeatureMean[k] * myFeatureMean[k]);
    float rawStd = sqrt(max(var, 0.0f));
    // Floor std at a fraction of the mean (not a tiny fixed epsilon) so
    // the anomaly-percent denominator can't collapse toward zero with a
    // small/noisy sample count.
    float minStd = max(myFeatureMean[k] * 0.15f, 0.01f);
    myFeatureStd[k] = max(rawStd, minStd);
  }
  myFeatureBaselineCount = n;
  myFeatureBaselineReady = true;

  const char* myFeatNames[FEATURES_PER_AXIS] = {"RMS", "P2P", "ZCR"};
  const char* myAxisNames[IMU_AXES] = {"X", "Y", "Z"};
  Serial.printf("\nSecondary feature baseline calibrated from %d normal samples:\n", n);
  for (int a = 0; a < IMU_AXES; a++)
    for (int fIdx = 0; fIdx < FEATURES_PER_AXIS; fIdx++) {
      int k = a * FEATURES_PER_AXIS + fIdx;
      Serial.printf("  %s-%s: mean=%.5f  std=%.5f\n",
                    myAxisNames[a], myFeatNames[fIdx], myFeatureMean[k], myFeatureStd[k]);
    }
  if (n < 8) Serial.println("WARNING: fewer than 8 normal samples - feature baseline will be noisy.");

  mySaveFeatureBaseline();
}

// Combines an axis's 3 feature z-scores into one 0-100% score (RMS
// combine - rises smoothly as multiple features drift together).
float myAxisZScoreToPercent(float featVec[TOTAL_FEATURES], int axis) {
  float sumSqZ = 0;
  for (int fIdx = 0; fIdx < FEATURES_PER_AXIS; fIdx++) {
    int k = axis * FEATURES_PER_AXIS + fIdx;
    float z = (featVec[k] - myFeatureMean[k]) / myFeatureStd[k];
    sumSqZ += z * z;
  }
  float combinedZ = sqrt(sumSqZ / FEATURES_PER_AXIS);
  return constrain((combinedZ / FEATURE_SIGMA) * 100.0f, 0.0f, 100.0f);
}

void myActionTrain() {
  if (!mySDavailable) {
    Serial.println("No SD - cannot train"); myResetMenuState(); return;
  }

  // Build training list
  myTrainingData.clear();
  int classCounts[NUM_CLASSES] = {};
  for (int c = 0; c < NUM_CLASSES; c++) {
    String path = "/motion/" + myClassLabels[c];
    File root = SD.open(path);
    if (!root) continue;
    while (File file = root.openNextFile()) {
      String name = file.name();
      if (!file.isDirectory() && name.endsWith(".csv")) {
        myTrainingData.push_back({path + "/" + name, c});
        classCounts[c]++;
      }
      file.close();
    }
    root.close();
  }

  Serial.println("\n=== Training ===");
  for (int c = 0; c < NUM_CLASSES; c++)
    Serial.printf("  %s: %d samples\n", myClassLabels[c].c_str(), classCounts[c]);

  // Shuffle and split validation
  std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
  int valCount = 0;
  std::vector<TrainingItem> myValData;
  if (VALIDATION_SAMPLES > 0) {
    // Hold out last VALIDATION_SAMPLES per class
    int heldOut[NUM_CLASSES] = {};
    std::vector<TrainingItem> trainOnly;
    for (auto& item : myTrainingData) {
      if (heldOut[item.label] < VALIDATION_SAMPLES) {
        myValData.push_back(item);
        heldOut[item.label]++;
        valCount++;
      } else {
        trainOnly.push_back(item);
      }
    }
    myTrainingData = trainOnly;
  }
  Serial.printf("Training: %d samples  Validation: %d samples\n",
                (int)myTrainingData.size(), valCount);

  if (myTrainingData.size() == 0) {
    Serial.println("No samples collected yet - use Collect for both classes first.");
    myResetMenuState();
    return;
  }
  for (int c = 0; c < NUM_CLASSES; c++) {
    if (classCounts[c] == 0) {
      Serial.printf("WARNING: no samples for class '%s' - collect some before training.\n",
                    myClassLabels[c].c_str());
    }
  }

  // Training-only buffers (grad/Adam) allocated here, freed before this
  // function returns - keeps steady-state PSRAM low (v002/v003 fix).
  if (!myAllocateTrainingMemory()) {
    Serial.println("FATAL: Could not allocate training memory - see allocation errors above.");
    myFreeTrainingMemory();
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "Train PSRAM"); u8g2.drawStr(0, 28, "alloc failed"); } while (u8g2.nextPage());
    delay(2500);
    myResetMenuState();
    return;
  }

  // Training loop
  float* myBatchBuf = (float*)ps_malloc(INPUT_SIZE * sizeof(float));
  if (!myCheckAlloc(myBatchBuf, "myBatchBuf", INPUT_SIZE * sizeof(float))) {
    myFreeTrainingMemory();
    myResetMenuState();
    return;
  }

  for (int epoch = 0; epoch < TARGET_EPOCHS; epoch++) {
    std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
    float epochLoss = 0;
    int   correct   = 0;
    int   processed = 0;
    myZeroGradients();

    for (int si = 0; si < (int)myTrainingData.size(); si++) {
      myCheckTouchBackground();  // keep touch responsive
      if (!myLoadSampleFromFile(myTrainingData[si].path.c_str(), myBatchBuf)) continue;

      myForwardPass(myBatchBuf);
      int label = myTrainingData[si].label;
      epochLoss += myComputeLoss(label);

      // Argmax for accuracy
      int pred = 0;
      for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
      if (pred == label) correct++;

      myBackwardPass(myBatchBuf, label);
      processed++;

      // Apply gradients at end of each mini-batch
      if ((si + 1) % BATCH_SIZE == 0 || si == (int)myTrainingData.size() - 1) {
        float scale = 1.0f / processed;
        for (int k = 0; k < CONV1_WEIGHTS;  k++) myConv1_w_grad[k]  *= scale;
        for (int k = 0; k < CONV1_FILTERS;  k++) myConv1_b_grad[k]  *= scale;
        for (int k = 0; k < DENSE1_WEIGHTS; k++) myDense1_w_grad[k] *= scale;
        for (int k = 0; k < DENSE1_SIZE;    k++) myDense1_b_grad[k] *= scale;
        for (int k = 0; k < DENSE2_WEIGHTS; k++) myDense2_w_grad[k] *= scale;
        for (int k = 0; k < DENSE2_SIZE;    k++) myDense2_b_grad[k] *= scale;
        for (int k = 0; k < OUTPUT_WEIGHTS; k++) myOutput_w_grad[k] *= scale;
        for (int k = 0; k < NUM_CLASSES;    k++) myOutput_b_grad[k] *= scale;

        myAdamUpdate(myConv1_w,  myConv1_w_grad,  myConv1_w_m,  myConv1_w_v,  CONV1_WEIGHTS,  LEARNING_RATE);
        myAdamUpdate(myConv1_b,  myConv1_b_grad,  myConv1_b_m,  myConv1_b_v,  CONV1_FILTERS,  LEARNING_RATE);
        myAdamUpdate(myDense1_w, myDense1_w_grad, myDense1_w_m, myDense1_w_v, DENSE1_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myDense1_b, myDense1_b_grad, myDense1_b_m, myDense1_b_v, DENSE1_SIZE,    LEARNING_RATE);
        myAdamUpdate(myDense2_w, myDense2_w_grad, myDense2_w_m, myDense2_w_v, DENSE2_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myDense2_b, myDense2_b_grad, myDense2_b_m, myDense2_b_v, DENSE2_SIZE,    LEARNING_RATE);
        myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES,    LEARNING_RATE);

        myZeroGradients();
        processed = 0;
      }
    }

    // Validation
    float valAcc = 0;
    if (valCount > 0) {
      int valCorrect = 0;
      for (auto& vi : myValData) {
        if (!myLoadSampleFromFile(vi.path.c_str(), myBatchBuf)) continue;
        myForwardPass(myBatchBuf);
        int pred = 0;
        for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
        if (pred == vi.label) valCorrect++;
      }
      valAcc = 100.0f * valCorrect / valCount;
    }

    float trainAcc = 100.0f * correct / max((int)myTrainingData.size(), 1);
    Serial.printf("Epoch %2d/%d  Loss=%.4f  TrainAcc=%.1f%%  ValAcc=%.1f%%\n",
                  epoch + 1, TARGET_EPOCHS,
                  epochLoss / max((int)myTrainingData.size(), 1),
                  trainAcc, valAcc);

    // OLED progress
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      char buf[24];
      snprintf(buf, sizeof(buf), "Ep %d/%d", epoch + 1, TARGET_EPOCHS);
      u8g2.drawStr(0, 8, buf);
      snprintf(buf, sizeof(buf), "Tr %.0f%%", trainAcc);
      u8g2.drawStr(0, 18, buf);
      if (valCount > 0) { snprintf(buf, sizeof(buf), "Val %.0f%%", valAcc); u8g2.drawStr(0, 28, buf); }
    } while (u8g2.nextPage());
  }

  free(myBatchBuf);
  myFreeTrainingMemory();   // release grad/Adam buffers - not needed again until next Train
  myWeightsTrained = true;
  mySaveWeights();
  Serial.println("ML weights saved.");

  // Also (re)compute the secondary per-axis statistical baseline from
  // the "1normal" class samples - this is what drives the RMS/P2P/ZCR
  // per-axis readout at Infer, independent of the ML model above.
  myComputeFeatureBaseline();

  u8g2.firstPage();
  do { u8g2.setFont(u8g2_font_5x7_tf); u8g2.drawStr(0, 15, "Training done!"); u8g2.drawStr(0, 28, "Weights saved"); } while (u8g2.nextPage());
  delay(2000);
  myResetMenuState();
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE FUNCTIONS                                             ██
// ██                                                                          ██
// ██  Continuously captures 1-second IMU windows and classifies them.         ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


void myActionInfer() {
  if (!myWeightsTrained) {
    Serial.println("No trained ML weights - train or load first");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No weights!"); u8g2.drawStr(0, 28, "Train first"); } while (u8g2.nextPage());
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Anomaly inference mode (tap A0 or 'l' to exit)");
  Serial.printf("ML anomaly = 100%% - P(%s).  Flag threshold=%.0f%%\n",
                myClassLabels[myNormalClassIdx].c_str(), ANOMALY_FLAG_PERCENT);
  if (!myFeatureBaselineReady) {
    Serial.println("NOTE: no secondary feature baseline yet - X/Y/Z readout will show 0%.");
    Serial.println("      Run Train again after collecting '1normal' samples to build it.");
  }

  float myLiveBuf[INPUT_SIZE];
  int   windowCount = 0;

  while (true) {
    if (myCheckTouchInput() == 2) { myResetMenuState(); return; }
    if (Serial.available()) { char c = Serial.read(); if (c == 'l' || c == 'L') { myResetMenuState(); return; } }

    // Capture one 1-second window
    for (int t = 0; t < IMU_TIMESTEPS; t++) {
      unsigned long tS = millis();
      myLiveBuf[t * IMU_AXES + 0] = myIMU.readFloatAccelX();
      myLiveBuf[t * IMU_AXES + 1] = myIMU.readFloatAccelY();
      myLiveBuf[t * IMU_AXES + 2] = myIMU.readFloatAccelZ();
      long el = millis() - tS;
      if (el < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - el);
    }

    // --- Secondary per-axis statistical readout (RMS/P2P/ZCR) ---
    // Extract BEFORE normalizing in-place, since myNormalizeInput()
    // divides by std and this readout needs raw amplitude preserved.
    float featBuf[INPUT_SIZE];
    memcpy(featBuf, myLiveBuf, sizeof(featBuf));
    myMeanCenterOnly(featBuf);
    float liveFeat[TOTAL_FEATURES];
    myExtractFeatures(featBuf, liveFeat);
    float xPct = myFeatureBaselineReady ? myAxisZScoreToPercent(liveFeat, 0) : 0.0f;
    float yPct = myFeatureBaselineReady ? myAxisZScoreToPercent(liveFeat, 1) : 0.0f;
    float zPct = myFeatureBaselineReady ? myAxisZScoreToPercent(liveFeat, 2) : 0.0f;

    // --- Primary ML confidence readout (overall anomaly %) ---
    myNormalizeInput(myLiveBuf);
    myForwardPass(myLiveBuf);
    float pNormal = myFinal_output[myNormalClassIdx];
    float mlAnomalyPct = constrain((1.0f - pNormal) * 100.0f, 0.0f, 100.0f);

    int rawPred = 0;
    for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[rawPred]) rawPred = j;

    windowCount++;
    bool isAnomaly = (mlAnomalyPct >= ANOMALY_FLAG_PERCENT) ||
                     (max(xPct, max(yPct, zPct)) >= ANOMALY_FLAG_PERCENT);

    Serial.printf("Win %d | ML anomaly=%.0f%% (P(%s)=%.0f%%, raw class=%s) | X=%.0f%% Y=%.0f%% Z=%.0f%% | [%s]\n",
                  windowCount, mlAnomalyPct, myClassLabels[myNormalClassIdx].c_str(), pNormal * 100,
                  myClassLabels[rawPred].c_str(), xPct, yPct, zPct, isAnomaly ? "ANOMALY" : "normal");

    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      char buf[20];
      snprintf(buf, sizeof(buf), "ML:%3.0f%% #%d", mlAnomalyPct, windowCount);
      u8g2.drawStr(0, 8, buf);
      snprintf(buf, sizeof(buf), "X%3.0f Y%3.0f Z%3.0f", xPct, yPct, zPct);
      u8g2.drawStr(0, 18, buf);
      u8g2.drawStr(0, 30, isAnomaly ? "** ANOMALY **" : "Status: normal");
    } while (u8g2.nextPage());
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM FUNCTIONS                                           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    String label =
      (i <= NUM_CLASSES) ? myClassLabels[i - 1] :
      (i == NUM_CLASSES + 1) ? "Train" : "Infer";
    Serial.printf("%s%d. %s\n", (i == myMenuIndex) ? " > " : "   ", i, label.c_str());
  }
  Serial.println("Commands: t=next  l=select");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    if (mySDavailable) {
      u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    } else {
      u8g2.drawStr(0, 8, "** NO SD CARD **");
    }
    int myStartItem = (myMenuIndex <= NUM_CLASSES) ? 1 : myMenuIndex - 2;
    for (int i = 0; i < 3; i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      String label =
        (cur <= NUM_CLASSES) ? myClassLabels[cur - 1] :
        (cur == NUM_CLASSES + 1) ? "Train" : "Infer";
      int y = 18 + i * 9;
      u8g2.drawStr(0, y, ((cur == myMenuIndex) ? "> " + label : "  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

void myExecuteMenuItem(int idx) {
  if      (idx <= NUM_CLASSES)        myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES + 1)    myActionTrain();
  else                                myActionInfer();
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();
    if (c >= '1' && c <= '9') {
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) {
        myMenuIndex = newIndex;
        myIsSelected = true;
        myLastActivityTime = myCurrentMillis;
        myExecuteMenuItem(myMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (c == 'l' || c == 'L') {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }

  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();
    if (touchAction == 1) {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (touchAction == 2) {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }
}

// ======================================================
// NOTE ON SENSOR FUSION EXTENSION
// ======================================================
// The 120-input vector is currently 40 × [ax, ay, az].
// To extend to other sensor combinations, change the layout here:
//
//   IMU_AXES = 6 → [ax, ay, az, gx, gy, gz]   40 × 6 = 240  (update INPUT_SIZE = 240)
//   IMU_AXES = 2 → [ax, ay]                    60 × 2 = 120  (adjust IMU_TIMESTEPS = 60)
//   Mixed sensors → concatenate channels in the buffer, one entry per timestep
//
// Per-channel normalization: extend myAccelMean[] and myAccelStd[] arrays
// to match the number of axes/channels being used and apply myNormalizeInput()
// channel by channel (or add a separate myNormalizeChannel() for non-accelerometer data).
//
// The dense network architecture (120 → 32 → 16 → NUM_CLASSES) does NOT change —
// only INPUT_SIZE and DENSE1_WEIGHTS need recomputing when the input vector changes.
// ======================================================