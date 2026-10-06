// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// MOTION ANOMALY DETECTION (2-CLASS ML + STATISTICAL FEATURES) — v002
// (pairs with index-anomaly-v002.html)
//
// This is your anomaly sketch (two classes, 0off and 1normal;
// ANOMALY % = 100% - P(1normal); plus a direct per-axis RMS / peak-to-peak /
// zero-crossing readout) with the web-page link of the motion v003 sketch:
//
//   0off    = device mounted on the machine, motor OFF
//   1normal = device mounted on the machine, motor running normally
//   2AnomalyTest (optional, never trained on) = recordings of a FAULT, used by
//             the page only to measure how well faults are detected
//
// Input: 40 samples x 3 axes (AccelX, AccelY, AccelZ) = 120 floats per window
// Sampling: ~40 Hz over ~1 second per capture window
//
// SD card stores: sensor recordings in folders /motion/<name>/sN.csv
// SD card stores: weights (.bin and .h), calibration, feature baseline, config.json
// Serial monitor and OLED output
//
// WHAT v002 ADDS TO YOUR ANOMALY SKETCH (every change is marked "// v002:"):
//   1. A link to the web page over WebBLE (phone or desktop) AND over Web
//      Serial (desktop). Both carry the SAME small frames, so the page can
//      collect, pull, push, train, infer and debug without moving the SD
//      card. The SD card stays the source of truth.
//   2. Layout is compile-time #defines (see ==LAYOUT START==) with
//      static_asserts and an exact weight-file size. A weights file of the
//      wrong size is REFUSED with a message that prints this sketch's layout.
//   3. The model package the page pushes or fetches carries the weights, the
//      calibration AND the feature baseline, so the ML score and the per-axis
//      readout always come from the same training data.
//   4. /header/config.json is read at boot: class names, and (runtime, no
//      recompile) "anomaly_flag_percent" and "feature_sigma". Layout numbers
//      are only COMPARED and a WARNING is printed on mismatch.
//   5. Every function and every global is declared before it is used (the
//      Arduino IDE needs that). No custom struct appears in a function
//      signature, so the IDE's automatic prototypes cannot trip.
//   6. Small fixes: a new sample never overwrites an older one with the same
//      number; training refuses to start unless BOTH classes have samples; the
//      validation hold-out always leaves at least one training sample per
//      class; max()/min() are not used on mixed types.
//
// NOT TESTED ON HARDWARE. It was compiled against an Arduino mock on a PC and
// its maths and link layer were run against the page's JavaScript.
// Bench-tune MY_SIGN_X/Y/Z so Z reads about +1 g when the board lies flat.
//
// LIBRARIES
//   Seeed Arduino LSM6DS3, U8g2, and (optional) NimBLE-Arduino 2.x by h2zero.
//   If you do not want BLE, put   #define MY_USE_BLE 0   before the includes.
//   Tools -> PSRAM: OPI PSRAM.
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
//            h2zero/NimBLE-Arduino @ ^2.x
// board_build.arduino.memory_type = qio_opi
//


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Globals, Declarations, Memory, Weights, Setup, Loop   ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// Optional: uncomment AFTER copying myAnomalyMLWeights.h from SD to sketch folder
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS

#ifndef MY_USE_BLE
  #define MY_USE_BLE 1            // v002: 1 = WebBLE link on, 0 = no NimBLE library needed
#endif

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
#include <stdarg.h>               // v002
#include <U8g2lib.h>
#include "mbedtls/base64.h"       // v002
#if MY_USE_BLE
  #include <NimBLEDevice.h>       // v002
#endif


// ======================================================
// v002: LAYOUT. Everything below derives from these #defines and they are
// COMPILE-TIME on purpose (buffers are sized from them). The web page shows
// the exact lines to paste here. NUM_CLASSES stays 2 (0off + 1normal).
// ==LAYOUT START==
#define NUM_CLASSES       2

#define IMU_TIMESTEPS     40
#define IMU_AXES           3      // AccelX, AccelY, AccelZ (fixed at 3 in this sketch)
#define INPUT_SIZE       (IMU_TIMESTEPS * IMU_AXES)   // 120
#define SAMPLE_INTERVAL_MS  25    // ~40 Hz => ~1 second window

// Conv1D -> MaxPool/2 -> Dense -> Dense -> Output
#define CONV1_KERNEL    5
#define CONV1_FILTERS   8
#define CONV1_OUT_STEPS (IMU_TIMESTEPS - CONV1_KERNEL + 1)   // 36
#define POOL1_STEPS     (CONV1_OUT_STEPS / 2)                // 18
#define CONV1_FLAT      (POOL1_STEPS * CONV1_FILTERS)        // 144
#define CONV1_WEIGHTS   (CONV1_KERNEL * IMU_AXES * CONV1_FILTERS)  // 120

#define DENSE1_SIZE   32
#define DENSE2_SIZE   16

#define DENSE1_WEIGHTS  (CONV1_FLAT  * DENSE1_SIZE)   // 4608
#define DENSE2_WEIGHTS  (DENSE1_SIZE * DENSE2_SIZE)   //  512
#define OUTPUT_WEIGHTS  (DENSE2_SIZE * NUM_CLASSES)   //   32

// Secondary per-axis readout: RMS, peak-to-peak and zero-crossing rate per axis
#define FEATURES_PER_AXIS  3
#define TOTAL_FEATURES     (IMU_AXES * FEATURES_PER_AXIS)   // 9

// Exact weights-file size (floats, little endian): conv w,b  dense1 w,b  dense2 w,b  output w,b
#define MY_WEIGHT_FLOATS (CONV1_WEIGHTS + CONV1_FILTERS + DENSE1_WEIGHTS + DENSE1_SIZE + \
                          DENSE2_WEIGHTS + DENSE2_SIZE + OUTPUT_WEIGHTS + NUM_CLASSES)   // 5331
#define MY_WEIGHT_BYTES  (MY_WEIGHT_FLOATS * 4)
// Feature baseline in the package: mean[9], std[9], count (as a float)
#define MY_FEAT_FLOATS   (TOTAL_FEATURES * 2 + 1)
// Package sent over the link = weights + calibration mean[3] std[3] + feature baseline
#define MY_PACKAGE_FLOATS (MY_WEIGHT_FLOATS + 2 * IMU_AXES + MY_FEAT_FLOATS)
#define MY_PACKAGE_BYTES  (MY_PACKAGE_FLOATS * 4)
// One buffer serves incoming blobs (model, sample, config.json) and outgoing ones
#define MY_BLOB_CAP ((MY_PACKAGE_BYTES) > 4200 ? (MY_PACKAGE_BYTES) : 4200)

static_assert(IMU_AXES == 3, "this sketch reads 3 accelerometer axes");
static_assert(CONV1_KERNEL >= 1, "kernel must be at least 1");
static_assert(CONV1_OUT_STEPS >= 2, "need at least 2 conv output steps for the 2x pool");
static_assert((CONV1_OUT_STEPS % 2) == 0, "IMU_TIMESTEPS - CONV1_KERNEL + 1 must be even (2x pool)");
static_assert(NUM_CLASSES == 2, "the anomaly sketch has exactly two classes: 0off and 1normal");
static_assert(INPUT_SIZE * 4 <= MY_BLOB_CAP, "a sample must fit the blob buffer");
// ==LAYOUT END==

#ifdef USE_BAKED_WEIGHTS
  // v002: refuse a baked-in header that was made for another layout
  static_assert(sizeof(myModel_conv1_w)  / sizeof(float) == CONV1_WEIGHTS,  "baked conv1_w size differs from this layout");
  static_assert(sizeof(myModel_dense1_w) / sizeof(float) == DENSE1_WEIGHTS, "baked dense1_w size differs from this layout");
  static_assert(sizeof(myModel_dense2_w) / sizeof(float) == DENSE2_WEIGHTS, "baked dense2_w size differs from this layout");
  static_assert(sizeof(myModel_output_w) / sizeof(float) == OUTPUT_WEIGHTS, "baked output_w size differs from this layout");
#endif

// v002: the folder list. 0 and 1 are the trained classes, the next one is the
// page-only test set (faults) that this sketch records but never trains on.
#define MY_TEST_FOLDER   "2AnomalyTest"
#define MY_DATA_FOLDERS  (NUM_CLASSES + 1)

// v002: one analysed window = this many floats (see myAnalyzeWindow)
#define MY_R_ML     0      // ML anomaly percent = 100 - P(normal) * 100
#define MY_R_PNORM  1      // P(1normal), 0..1
#define MY_R_X      2      // per-axis anomaly percent
#define MY_R_Y      3
#define MY_R_Z      4
#define MY_R_FLAG   5      // 1 = ANOMALY overall, else 0
#define MY_R_PRED   6      // argmax class of the classifier
#define MY_R_FEAT   7      // then TOTAL_FEATURES live features
#define MY_RES_N    (MY_R_FEAT + TOTAL_FEATURES)


// ======================================================
// CONFIGURATION & ML HYPERPARAMETERS
// ======================================================
String myClassLabels[NUM_CLASSES] = {"0off", "1normal"};
const int myNormalClassIdx = 1;   // index of "1normal" - the class we score confidence against

const int myTotalItems = NUM_CLASSES + 2;  // classes + Train + Infer

float LEARNING_RATE  = 0.001f;
int   BATCH_SIZE     = 6;
int   TARGET_EPOCHS  = 30;
int   VALIDATION_SAMPLES = 3;   // up to N samples per class held out for validation (0 = disabled)

// If ANY axis anomaly percent (from the statistical readout) is >= this, OR the
// ML anomaly percent is >= this, the window is flagged ANOMALY.
// v002: config.json may override it at run time ("anomaly_flag_percent").
float ANOMALY_FLAG_PERCENT = 50.0f;

// How many standard deviations away from the normal baseline counts as 100%
// anomalous for the per-axis readout. v002: config.json "feature_sigma".
float FEATURE_SIGMA = 3.0f;

// ======================================================
// v002: SENSOR PARITY. The page produces windows in the SAME frame:
// units of g (gravity included), order ax,ay,az, 25 ms apart.
// If your board's axes come out mirrored, flip the sign here (and tell the
// page's phone mapping the same). Bench-tune: Z should be about +1 when flat.
// Changing a sign makes older recordings incompatible: recapture them.
// ======================================================
#define MY_SIGN_X  1.0f
#define MY_SIGN_Y  1.0f
#define MY_SIGN_Z  1.0f

// ======================================================
// SECONDARY PER-AXIS READOUT: RMS / peak-to-peak / zero-crossing-rate per axis,
// computed directly (no training). Baseline = mean and std of these features
// over the "1normal" recordings, rebuilt at the end of every Train.
// ======================================================
enum { FEAT_RMS = 0, FEAT_P2P = 1, FEAT_ZCR = 2 };
float myFeatureMean[TOTAL_FEATURES] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
float myFeatureStd [TOTAL_FEATURES] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
int   myFeatureBaselineCount = 0;
bool  myFeatureBaselineReady = false;

// ======================================================
// NORMALIZATION CONSTANTS (computed at startup by myCalibrate())
// Defaults used only if calibration is skipped.
// Z axis default mean=1.0 accounts for gravity when device is flat.
// QUIRK (kept on purpose, the page copies it): std comes from a STILL window and is
// floored at 0.01, so real motion is many sigma and is clipped at +-5.
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
// DEVICES
// ======================================================
LSM6DS3 myIMU(I2C_MODE, 0x6A);    // IMU object using I2C interface
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// ML WEIGHT & GRADIENT BUFFERS (PSRAM)
// ======================================================
float* myInputBuffer = nullptr;   // INPUT_SIZE floats per inference/training step

// Conv1D weights and biases
float* myConv1_w = nullptr;       // CONV1_WEIGHTS = kernel x axes x filters
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
float* myConv1_output  = nullptr;   // CONV1_OUT_STEPS x CONV1_FILTERS (pre-pool)
float* myPool1_output  = nullptr;   // POOL1_STEPS     x CONV1_FILTERS = CONV1_FLAT (post-pool)
float* myDense1_output = nullptr;   // DENSE1_SIZE
float* myDense2_output = nullptr;   // DENSE2_SIZE
float* myFinal_output  = nullptr;   // NUM_CLASSES  (softmax probabilities)

// Backward-pass delta buffers
float* myOutput_delta = nullptr;   // NUM_CLASSES
float* myDense2_delta = nullptr;   // DENSE2_SIZE
float* myDense1_delta = nullptr;   // DENSE1_SIZE
float* myPool1_delta  = nullptr;   // CONV1_FLAT
float* myConv1_delta  = nullptr;   // CONV1_OUT_STEPS x CONV1_FILTERS

// Adam step counter (QUIRK kept: it counts once per parameter ARRAY update)
int myAdamStep = 0;

struct TrainingItem {
  String path;
  int    label;
};
std::vector<TrainingItem> myTrainingData;
bool myCoreMemoryReady = false;     // v002: weights + forward buffers exist (gradients/Adam only exist while training)


// ======================================================
// v002: LINK STATE (BLE + Web Serial share the same frames)
// Frame = [type][payload]. Types: 'H' blob header, 'D' blob chunk,
// 'A' ack, 'T' text command (page -> device), 'R' text reply (device -> page).
// BLE carries a frame per write/notify. Serial carries "@B <base64(frame)>\n".
// A "blob" (model, sample, config.json) is sent as H, then chunks of
// MY_CHUNK_DATA bytes, acked every MY_WIN chunks with the next expected chunk.
// ==LINK VARS START==
#define MY_DEVICE_NAME  "XIAO-Anomaly-01"
#define MY_SVC_UUID     "7e700001-b2c3-5d4e-af60-9b3c7d8eaf20"
#define MY_CMD_UUID     "7e700002-b2c3-5d4e-af60-9b3c7d8eaf20"   // page -> device, write with response
#define MY_EVT_UUID     "7e700003-b2c3-5d4e-af60-9b3c7d8eaf20"   // device -> page, notify

#define MY_CHUNK_DATA   160
#define MY_FRAME_MAX    (MY_CHUNK_DATA + 8)
#define MY_WIN          4
#define MY_ACK_TIMEOUT_MS 2500
#define MY_QN           12
#define MY_F_HEAD  'H'
#define MY_F_DATA  'D'
#define MY_F_ACK   'A'
#define MY_F_TEXT  'T'
#define MY_F_RESP  'R'

struct MyFrame {
  uint8_t len;
  uint8_t via;                  // 1 = BLE, 2 = Serial
  uint8_t d[MY_FRAME_MAX];
};
MyFrame myQ[MY_QN];             // BLE callback -> loop() queue
volatile uint8_t myQHead = 0;
volatile uint8_t myQTail = 0;

volatile uint8_t myReplyVia = 0;        // where replies go: 0 nowhere, 1 BLE, 2 Serial
volatile bool    myBleConnected = false;
bool             myLinkDebugOn = false;
unsigned long    myDebugLastMs = 0;
unsigned long    myLastHbMs = 0;
volatile bool    myBusy = false;
volatile bool    myStopRequested = false;
uint8_t*         myBlobBuf = nullptr;   // MY_BLOB_CAP bytes in PSRAM

volatile bool    myRxActive = false;    // a blob is arriving
uint8_t          myRxKind = 0;
uint8_t          myRxId = 0;
uint32_t         myRxTotal = 0;
uint32_t         myRxGot = 0;
uint32_t         myRxCrc = 0;
uint16_t         myRxNext = 0;
unsigned long    myRxLastMs = 0;

volatile bool    mySending = false;     // inside mySendBlob(): once its ack arrives, stop reading so the NEXT command is handled after this one finishes
volatile uint16_t myTxAckNext = 0xFFFF; // last ack from the page (0xFFFF = none yet)
volatile uint8_t  myTxAckStatus = 0;    // 0 ok, 1 crc error, 2 abort

char             mySerLine[240];        // one "@B ..." line from Web Serial
int              mySerLen = 0;
// ==LINK VARS END==

#if MY_USE_BLE
NimBLEServer*         myBleServer = nullptr;
NimBLECharacteristic* myCmdChar   = nullptr;
NimBLECharacteristic* myEvtChar   = nullptr;
#endif

#define MY_NAME_MAX 32
bool myLinkWasDebug = false;

// ======================================================
// UTILITY FUNCTIONS (defined before use)
// ======================================================
inline float myClip(float v, float mn=-100, float mx=100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}

inline float myLeakyRelu(float x)      { return x > 0 ? x : 0.1f * x; }
inline float myLeakyReluDeriv(float x) { return x > 0 ? 1.0f : 0.1f; }


// v002: written without max()/min() so it compiles whether those are macros or templates
inline float myMaxF(float a, float b) { return a > b ? a : b; }
inline int   myMaxI(int a, int b)     { return a > b ? a : b; }



// ======================================================
// v002: FORWARD DECLARATIONS of every function (Arduino IDE friendly).
// ======================================================
String myFolderName(int idx);
void  mySoftmax(float* x, int size);
void  myNormalizeInput(float* buf);
void  myReadAccel(float* v);
bool  myKeyAvailable();
char  myKeyRead();
int   myReadTouch();
void  myResetTouchState();
void  myUpdateTouchState();
int   myCheckTouchInput();
void  myCheckTouchBackground();
bool  myCheckAlloc(void* ptr, const char* name, size_t bytes);
bool  myAllocateCoreMemory();
bool  myAllocateTrainingMemory();
void  myFreeTrainingMemory();
void  myPrintLayout();
void  myExportHeader();
bool  myLoadWeights();
void  mySaveWeights();
void  mySaveCalib();
void  myPackToBuf(float* out);
bool  myPackFromBuf(const float* in);
bool  myLoadFeatureBaseline();
void  mySaveFeatureBaseline();
bool  myComputeFeatureBaseline();
void  myMeanCenterOnly(float* buf);
void  myExtractFeatures(const float* window, float* featOut);
bool  myLoadAndExtractFeatures(const char* path, float* featOut);
float myAxisZScoreToPercent(const float* featVec, int axis);
void  myAnalyzeWindow(const float* raw, float* res);
int   myCfgFindKey(const char* t, const char* key);
bool  myCfgInt(const char* t, const char* key, int* out);
bool  myCfgFloat(const char* t, const char* key, float* out);
int   myCfgClasses(const char* t, char names[][MY_NAME_MAX], int maxN);
void  myApplyConfig(const char* t);
void  myLoadConfigFromSD();
void  myCalibrate(bool force);
int   myCountSamples(int classIdx);
bool  myNewSamplePath(int classIdx, String* out);
bool  myNthSamplePath(int classIdx, int n, String* out);
void  myCaptureWindow(float* w, bool echo);
bool  myWriteSample(int classIdx, const float* w, String* pathOut);
bool  myCaptureSample(int classIdx);
void  myActionCollect(int classIdx);
void  myConv1DForward(float* input);
void  myPool1Forward();
void  myDenseForward(float* input, int inSize, float* w, float* b, float* output, int outSize, bool applyActivation);
void  myForwardPass(float* input);
float myComputeLoss(int label);
void  myAdamUpdate(float* w, float* grad, float* m, float* v, int size, float lr);
void  myZeroGradients();
void  myBackwardPass(float* input, int label);
bool  myReadSampleCsv(const char* path, float* buf);
bool  myLoadSampleFromFile(const char* path, float* buf);
bool  myTrainCore();
void  myActionTrain();
void  myActionInfer();
void  myResetMenuState();
void  myDrawMenu();
void  myExecuteMenuItem(int idx);
void  myHandleMenuNavigation();

uint32_t myCrc32(const uint8_t* d, size_t n);
void  myPut32(uint8_t* p, uint32_t v);
uint32_t myGet32(const uint8_t* p);
bool  mySendFrame(const uint8_t* d, size_t n);
void  myReply(const char* fmt, ...);
void  myAck(uint16_t next, uint8_t status);
bool  mySendBlob(uint8_t kind, uint8_t id, const uint8_t* data, uint32_t total);
void  myOnHead(const uint8_t* d);
void  myOnData(const uint8_t* d, size_t n);
void  myHandleFrame(const uint8_t* d, size_t n);
void  myQPush(const uint8_t* d, size_t n, uint8_t via);
void  myPollSerialFrames();
void  myPumpIncoming();

void  myHandleCommand(char* s);
void  myDispatchBlob(uint8_t kind, uint8_t id, uint32_t total);
void  myImportPackage();
void  myStoreSample(int classIdx);
void  myStoreConfig(uint32_t total);
void  myReplyInfo();
void  myReplyCal();
void  myLinkHeartbeat();
#if MY_USE_BLE
void  myStartBle();
#endif


// ======================================================
// INPUT HELPERS
// ======================================================
// v002: folder name of a data folder: the two class labels, then the page-only test set
String myFolderName(int idx) {
  if (idx >= 0 && idx < NUM_CLASSES) return myClassLabels[idx];
  return String(MY_TEST_FOLDER);
}


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

// v002: ONE place that reads the IMU, in g, with the sign settings above.
void myReadAccel(float* v) {
  v[0] = MY_SIGN_X * myIMU.readFloatAccelX();
  v[1] = MY_SIGN_Y * myIMU.readFloatAccelY();
  v[2] = MY_SIGN_Z * myIMU.readFloatAccelZ();
}

// v002: all single-key input goes through these, so "@B ..." frame lines
// from the page are consumed first and never reach the menu.
bool myKeyAvailable() {
  myPollSerialFrames();
  return Serial.available() > 0;
}
char myKeyRead() {
  return (char)Serial.read();
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

// CORE allocation: weights + forward-pass buffers (+ the link's blob buffer).
// Everything Collect / Infer / boot ever needs. Stays allocated for program life.
bool myAllocateCoreMemory() {
  if (myCoreMemoryReady) return true;
  Serial.println("\n=== Allocating Core Memory (weights + forward buffers) ===");
  Serial.printf("Free PSRAM before allocation: %d bytes\n", ESP.getFreePsram());

  bool ok = true;
  #define MY_ALLOC_CORE(ptr, size, name) \
    ptr = (float*)ps_malloc((size) * sizeof(float)); \
    ok &= myCheckAlloc(ptr, name, (size) * sizeof(float));

  MY_ALLOC_CORE(myInputBuffer, INPUT_SIZE, "myInputBuffer");
  myBlobBuf = (uint8_t*)ps_malloc(MY_BLOB_CAP);                     // v002
  ok &= myCheckAlloc(myBlobBuf, "myBlobBuf", MY_BLOB_CAP);

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

  // He initialization (the constants derive from the layout #defines)
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

// TRAINING-ONLY allocation: gradients + Adam m/v + backward deltas. Allocated at
// the start of Train, freed right after - keeps steady-state PSRAM use low.
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

// v002: prints what THIS sketch was compiled for (used in refusal messages)
void myPrintLayout() {
  Serial.printf("Sketch layout: NUM_CLASSES=%d IMU_TIMESTEPS=%d IMU_AXES=%d CONV1_KERNEL=%d CONV1_FILTERS=%d "
                "DENSE1_SIZE=%d DENSE2_SIZE=%d SAMPLE_INTERVAL_MS=%d -> %d weight floats (%d bytes)\n",
                NUM_CLASSES, IMU_TIMESTEPS, IMU_AXES, CONV1_KERNEL, CONV1_FILTERS,
                DENSE1_SIZE, DENSE2_SIZE, SAMPLE_INTERVAL_MS, MY_WEIGHT_FLOATS, MY_WEIGHT_BYTES);
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
  // v002: refuse a file made for another layout instead of loading garbage
  if ((size_t)f.size() != (size_t)MY_WEIGHT_BYTES) {
    Serial.printf("REFUSED: myAnomalyMLWeights.bin is %u bytes but this sketch needs %u bytes.\n",
                  (unsigned)f.size(), (unsigned)MY_WEIGHT_BYTES);
    myPrintLayout();
    f.close();
    return false;
  }
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

// v002: split out of myCalibrate() so a model pushed from the page can save its calibration too
void mySaveCalib() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myCalib.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myAccelMean, IMU_AXES * 4);
    f.write((uint8_t*)myAccelStd,  IMU_AXES * 4);
    f.close();
    Serial.println("Calibration saved to SD");
  }
}

// v002: weights in file order, then calibration, then the feature baseline
// (mean[9], std[9], count as a float). Used for the link package.
void myPackToBuf(float* out) {
  float* src[8]  = { myConv1_w, myConv1_b, myDense1_w, myDense1_b, myDense2_w, myDense2_b, myOutput_w, myOutput_b };
  int    cnt[8]  = { CONV1_WEIGHTS, CONV1_FILTERS, DENSE1_WEIGHTS, DENSE1_SIZE, DENSE2_WEIGHTS, DENSE2_SIZE, OUTPUT_WEIGHTS, NUM_CLASSES };
  int o = 0;
  for (int b = 0; b < 8; b++) { memcpy(out + o, src[b], cnt[b] * 4); o += cnt[b]; }
  memcpy(out + o, myAccelMean, IMU_AXES * 4); o += IMU_AXES;
  memcpy(out + o, myAccelStd,  IMU_AXES * 4); o += IMU_AXES;
  memcpy(out + o, myFeatureMean, TOTAL_FEATURES * 4); o += TOTAL_FEATURES;
  memcpy(out + o, myFeatureStd,  TOTAL_FEATURES * 4); o += TOTAL_FEATURES;
  out[o] = myFeatureBaselineReady ? (float)myFeatureBaselineCount : 0.0f;
}

// v002: returns false (and changes nothing) if the package holds NaN or Infinity
bool myPackFromBuf(const float* in) {
  for (int i = 0; i < MY_PACKAGE_FLOATS; i++) if (isnan(in[i]) || isinf(in[i])) return false;
  float* dst[8]  = { myConv1_w, myConv1_b, myDense1_w, myDense1_b, myDense2_w, myDense2_b, myOutput_w, myOutput_b };
  int    cnt[8]  = { CONV1_WEIGHTS, CONV1_FILTERS, DENSE1_WEIGHTS, DENSE1_SIZE, DENSE2_WEIGHTS, DENSE2_SIZE, OUTPUT_WEIGHTS, NUM_CLASSES };
  int o = 0;
  for (int b = 0; b < 8; b++) { memcpy(dst[b], in + o, cnt[b] * 4); o += cnt[b]; }
  memcpy(myAccelMean, in + o, IMU_AXES * 4); o += IMU_AXES;
  memcpy(myAccelStd,  in + o, IMU_AXES * 4); o += IMU_AXES;
  memcpy(myFeatureMean, in + o, TOTAL_FEATURES * 4); o += TOTAL_FEATURES;
  memcpy(myFeatureStd,  in + o, TOTAL_FEATURES * 4); o += TOTAL_FEATURES;
  myFeatureBaselineCount = (int)(in[o] + 0.5f);
  myFeatureBaselineReady = (myFeatureBaselineCount > 0);
  return true;
}


// ======================================================
// v002: SECONDARY PER-AXIS READOUT (from your anomaly sketch): RMS / peak-to-peak /
// zero-crossing-rate per axis, computed directly - no training needed.
// This is what gives per-axis granularity, since the Conv1D classifier only
// produces one combined confidence number.
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

void myExtractFeatures(const float* window, float* featOut) {
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

// v002: reads a recording with the shared csv reader (stack only, no PSRAM)
bool myLoadAndExtractFeatures(const char* path, float* featOut) {
  float buf[INPUT_SIZE];
  if (!myReadSampleCsv(path, buf)) return false;
  myMeanCenterOnly(buf);
  myExtractFeatures(buf, featOut);
  return true;
}

bool myLoadFeatureBaseline() {
  if (!mySDavailable) return false;
  if (!SD.exists("/header/myFeatureBaseline.bin")) return false;
  File f = SD.open("/header/myFeatureBaseline.bin", FILE_READ);
  if (!f) return false;
  if ((size_t)f.size() != (size_t)(TOTAL_FEATURES * 8 + sizeof(int))) {       // v002: refuse another layout
    Serial.println("myFeatureBaseline.bin has the wrong size - ignored");
    f.close();
    return false;
  }
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

// Computes the RMS/P2P/ZCR baseline from ONLY the "1normal" class captures
// (the "0off" captures are not representative of running vibration).
// v002: returns true if a baseline was computed.
bool myComputeFeatureBaseline() {
  if (!mySDavailable) return false;
  String path = "/motion/" + myClassLabels[myNormalClassIdx];
  File root = SD.open(path);
  if (!root) { Serial.println("No normal-class folder found for feature baseline."); return false; }

  float sum[TOTAL_FEATURES]  = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float sum2[TOTAL_FEATURES] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int n = 0;

  while (File file = root.openNextFile()) {
    String name = file.name();
    bool isCsv = !file.isDirectory() && name.endsWith(".csv");
    file.close();
    if (!isCsv) continue;
    myCheckTouchBackground();
    myPumpIncoming();                                                   // v002
    float feat[TOTAL_FEATURES];
    if (!myLoadAndExtractFeatures((path + "/" + name).c_str(), feat)) continue;
    for (int k = 0; k < TOTAL_FEATURES; k++) { sum[k] += feat[k]; sum2[k] += feat[k] * feat[k]; }
    n++;
  }
  root.close();

  if (n == 0) { Serial.println("No normal-class samples - feature baseline not updated."); return false; }

  for (int k = 0; k < TOTAL_FEATURES; k++) {
    myFeatureMean[k] = sum[k] / n;
    float var = (sum2[k] / n) - (myFeatureMean[k] * myFeatureMean[k]);
    float rawStd = sqrt(myMaxF(var, 0.0f));
    // Floor std at a fraction of the mean (not a tiny fixed epsilon) so the
    // anomaly-percent denominator cannot collapse with a small/noisy sample count.
    float minStd = myMaxF(myFeatureMean[k] * 0.15f, 0.01f);
    myFeatureStd[k] = myMaxF(rawStd, minStd);
  }
  myFeatureBaselineCount = n;
  myFeatureBaselineReady = true;

  const char* myFeatNames[FEATURES_PER_AXIS] = {"RMS", "P2P", "ZCR"};
  const char* myAxisNames[IMU_AXES] = {"X", "Y", "Z"};
  Serial.printf("\nFeature baseline calibrated from %d normal samples:\n", n);
  for (int a = 0; a < IMU_AXES; a++)
    for (int fIdx = 0; fIdx < FEATURES_PER_AXIS; fIdx++) {
      int k = a * FEATURES_PER_AXIS + fIdx;
      Serial.printf("  %s-%s: mean=%.5f  std=%.5f\n",
                    myAxisNames[a], myFeatNames[fIdx], myFeatureMean[k], myFeatureStd[k]);
    }
  if (n < 8) Serial.println("WARNING: fewer than 8 normal samples - feature baseline will be noisy.");

  mySaveFeatureBaseline();
  return true;
}

// Combines an axis's 3 feature z-scores into one 0-100% score (RMS
// combine - rises smoothly as multiple features drift together).
float myAxisZScoreToPercent(const float* featVec, int axis) {
  float sumSqZ = 0;
  for (int fIdx = 0; fIdx < FEATURES_PER_AXIS; fIdx++) {
    int k = axis * FEATURES_PER_AXIS + fIdx;
    float z = (featVec[k] - myFeatureMean[k]) / myFeatureStd[k];
    sumSqZ += z * z;
  }
  float combinedZ = sqrt(sumSqZ / FEATURES_PER_AXIS);
  return constrain((combinedZ / FEATURE_SIGMA) * 100.0f, 0.0f, 100.0f);
}

// v002: ONE analysis used by the menu's Infer, by the page's INFER command and
// mirrored line by line in the page. raw = one window in g (not normalized).
// res gets MY_RES_N floats (see MY_R_*). Leaves the softmax in myFinal_output.
void myAnalyzeWindow(const float* raw, float* res) {
  // Per-axis statistical readout on the mean-centred RAW window (amplitude must survive)
  float featBuf[INPUT_SIZE];
  memcpy(featBuf, raw, sizeof(featBuf));
  myMeanCenterOnly(featBuf);
  myExtractFeatures(featBuf, res + MY_R_FEAT);
  float xp = myFeatureBaselineReady ? myAxisZScoreToPercent(res + MY_R_FEAT, 0) : 0.0f;
  float yp = myFeatureBaselineReady ? myAxisZScoreToPercent(res + MY_R_FEAT, 1) : 0.0f;
  float zp = myFeatureBaselineReady ? myAxisZScoreToPercent(res + MY_R_FEAT, 2) : 0.0f;

  // Primary ML readout: overall anomaly percent = 100% - P(normal)
  float x[INPUT_SIZE];
  memcpy(x, raw, sizeof(x));
  myNormalizeInput(x);
  myForwardPass(x);
  float pNormal = myFinal_output[myNormalClassIdx];
  float mlPct = constrain((1.0f - pNormal) * 100.0f, 0.0f, 100.0f);
  int pred = 0;
  for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;

  res[MY_R_ML] = mlPct;
  res[MY_R_PNORM] = pNormal;
  res[MY_R_X] = xp; res[MY_R_Y] = yp; res[MY_R_Z] = zp;
  res[MY_R_FLAG] = (mlPct >= ANOMALY_FLAG_PERCENT || myMaxF(xp, myMaxF(yp, zp)) >= ANOMALY_FLAG_PERCENT) ? 1.0f : 0.0f;
  res[MY_R_PRED] = (float)pred;
}


// ======================================================
// v002: CONFIG.JSON  (tiny hand-written parser, no JSON library)
// "classes" and the two anomaly numbers are USED, the layout numbers are only COMPARED.

// ==CFG PARSE START==
// Returns the index just after  "key" :  (skipping spaces), or -1 if the key is missing.
int myCfgFindKey(const char* t, const char* key) {
  char pat[32];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char* p = strstr(t, pat);
  if (!p) return -1;
  p += strlen(pat);
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  if (*p != ':') return -1;
  p++;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  return (int)(p - t);
}

bool myCfgInt(const char* t, const char* key, int* out) {
  int i = myCfgFindKey(t, key);
  if (i < 0) return false;
  *out = atoi(t + i);
  return true;
}

// Reads  "classes": ["a","b",...]  Returns how many strings the list holds
// (stores at most maxN of them), or -1 if there is no list.
int myCfgClasses(const char* t, char names[][MY_NAME_MAX], int maxN) {
  int i = myCfgFindKey(t, "classes");
  if (i < 0 || t[i] != '[') return -1;
  const char* p = t + i + 1;
  int count = 0;
  while (*p && *p != ']') {
    if (*p == '"') {
      p++;
      int n = 0;
      char tmp[MY_NAME_MAX];
      while (*p && *p != '"') { if (n < MY_NAME_MAX - 1) tmp[n++] = *p; p++; }
      tmp[n] = 0;
      if (*p == '"') p++;
      if (count < maxN) strcpy(names[count], tmp);
      count++;
    } else {
      p++;
    }
  }
  return count;
}
// ==CFG PARSE END==

// v002: reads  "key": 12.5   (also negative and exponent forms)
bool myCfgFloat(const char* t, const char* key, float* out) {
  int i = myCfgFindKey(t, key);
  if (i < 0) return false;
  *out = (float)atof(t + i);
  return true;
}

void myApplyConfig(const char* t) {
  char names[NUM_CLASSES + 1][MY_NAME_MAX];
  int n = myCfgClasses(t, names, NUM_CLASSES + 1);
  if (n == NUM_CLASSES) {
    for (int i = 0; i < NUM_CLASSES; i++) myClassLabels[i] = String(names[i]);
    Serial.println("config.json: class labels loaded");
  } else if (n >= 0) {
    Serial.printf("config.json: lists %d classes but this sketch has NUM_CLASSES=%d - keeping compiled labels\n", n, NUM_CLASSES);
  } else {
    Serial.println("config.json: no \"classes\" list - keeping compiled labels");
  }
  float fv;
  if (myCfgFloat(t, "anomaly_flag_percent", &fv)) {
    if (fv >= 1.0f && fv <= 100.0f) { ANOMALY_FLAG_PERCENT = fv; Serial.printf("config.json: anomaly flag percent = %.1f\n", fv); }
    else Serial.println("config.json: anomaly_flag_percent must be 1..100 - ignored");
  }
  if (myCfgFloat(t, "feature_sigma", &fv)) {
    if (fv >= 0.5f && fv <= 20.0f) { FEATURE_SIGMA = fv; Serial.printf("config.json: feature sigma = %.2f\n", fv); }
    else Serial.println("config.json: feature_sigma must be 0.5..20 - ignored");
  }
  int v;
  if (myCfgInt(t, "timesteps",   &v) && v != IMU_TIMESTEPS)      Serial.printf("WARNING: config timesteps=%d but sketch IMU_TIMESTEPS=%d\n", v, IMU_TIMESTEPS);
  if (myCfgInt(t, "axes",        &v) && v != IMU_AXES)           Serial.printf("WARNING: config axes=%d but sketch IMU_AXES=%d\n", v, IMU_AXES);
  if (myCfgInt(t, "kernel",      &v) && v != CONV1_KERNEL)       Serial.printf("WARNING: config kernel=%d but sketch CONV1_KERNEL=%d\n", v, CONV1_KERNEL);
  if (myCfgInt(t, "filters",     &v) && v != CONV1_FILTERS)      Serial.printf("WARNING: config filters=%d but sketch CONV1_FILTERS=%d\n", v, CONV1_FILTERS);
  if (myCfgInt(t, "dense1",      &v) && v != DENSE1_SIZE)        Serial.printf("WARNING: config dense1=%d but sketch DENSE1_SIZE=%d\n", v, DENSE1_SIZE);
  if (myCfgInt(t, "dense2",      &v) && v != DENSE2_SIZE)        Serial.printf("WARNING: config dense2=%d but sketch DENSE2_SIZE=%d\n", v, DENSE2_SIZE);
  if (myCfgInt(t, "interval_ms", &v) && v != SAMPLE_INTERVAL_MS) Serial.printf("WARNING: config interval_ms=%d but sketch SAMPLE_INTERVAL_MS=%d\n", v, SAMPLE_INTERVAL_MS);
  if (myCfgInt(t, "input_size",  &v) && v != INPUT_SIZE)         Serial.printf("WARNING: config input_size=%d but sketch INPUT_SIZE=%d\n", v, INPUT_SIZE);
}

void myLoadConfigFromSD() {
  if (!mySDavailable || !myBlobBuf) return;
  if (!SD.exists("/header/config.json")) { Serial.println("No /header/config.json - using compiled class labels"); return; }
  File f = SD.open("/header/config.json", FILE_READ);
  if (!f) return;
  size_t n = f.size();
  if (n == 0 || n > 4096) { Serial.println("config.json: empty or larger than 4 KB - ignored"); f.close(); return; }
  f.read(myBlobBuf, n);
  f.close();
  myBlobBuf[n] = 0;
  myApplyConfig((const char*)myBlobBuf);
}



// ======================================================
// CALIBRATION  (run once at startup, device stationary)
// Collects CALIB_SAMPLES readings and computes per-axis mean and std.
// Saves result to SD so subsequent boots skip the wait.
// v002: force=true recalibrates even if a saved file exists (page button).
// ======================================================
void myCalibrate(bool force) {
  // Try loading from SD first
  if (!force && mySDavailable && SD.exists("/header/myCalib.bin")) {
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
    myReadAccel(v);                                    // v002
    for (int a = 0; a < IMU_AXES; a++) { sum[a] += v[a]; sum2[a] += v[a] * v[a]; }
    delay(SAMPLE_INTERVAL_MS);
  }

  for (int a = 0; a < IMU_AXES; a++) {
    myAccelMean[a] = sum[a] / CALIB_SAMPLES;
    float var = (sum2[a] / CALIB_SAMPLES) - (myAccelMean[a] * myAccelMean[a]);
    float sd = sqrt(var);                      // v002: written without max() so it compiles whether max is a macro or a template
    myAccelStd[a]  = (sd > 0.01f) ? sd : 0.01f;   // floor at 0.01 to avoid div/0
  }

  Serial.printf("Calibration done: mean=%.3f,%.3f,%.3f  std=%.3f,%.3f,%.3f\n",
                myAccelMean[0], myAccelMean[1], myAccelMean[2],
                myAccelStd[0],  myAccelStd[1],  myAccelStd[2]);

  mySaveCalib();                                       // v002 (was inline)
}


// ======================================================
// SETUP AND LOOP
// ======================================================
void setup() {
  Serial.setRxBufferSize(1024);   // v002: room for a few "@B ..." frame lines (comment out if your core has no such call)
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32-S3 Motion Anomaly System v002 Starting ===");
  Serial.printf("Free heap:  %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  myPrintLayout();                                     // v002

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
    Serial.println("  (USE_BAKED_WEIGHTS) are compiled in or a model is");
    Serial.println("  pushed from the page.");
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
  myCalibrate(false);

  if (!myAllocateCoreMemory()) {
    Serial.println("FATAL: Core PSRAM allocation failed - see allocation errors above.");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 12, "PSRAM ERROR!"); u8g2.drawStr(0, 26, "See Serial log"); } while (u8g2.nextPage());
    while (1) { delay(1000); }
  }
  myLoadConfigFromSD();                                // v002 (needs the blob buffer)

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

  myLoadFeatureBaseline();   // per-axis RMS/P2P/ZCR baseline (optional, not fatal if missing)

#if MY_USE_BLE
  myStartBle();                                        // v002
#endif

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);
  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() {
  myPumpIncoming();             // v002: frames from the page (BLE queue + Web Serial lines)
  myHandleMenuNavigation();
  myLinkHeartbeat();            // v002: live ax,ay,az for the page while "debug frames" is on
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
  String path = "/motion/" + myFolderName(classIdx);
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

// v002: first free file name sN.csv (v001 used the file COUNT, which could
// overwrite an existing file after a deletion).
bool myNewSamplePath(int classIdx, String* out) {
  if (!mySDavailable) return false;
  String folderPath = "/motion/" + myFolderName(classIdx);
  if (!SD.exists("/motion")) SD.mkdir("/motion");
  if (!SD.exists(folderPath)) SD.mkdir(folderPath);
  for (int n = myCountSamples(classIdx); n < 100000; n++) {
    String p = folderPath + "/s" + String(n) + ".csv";
    if (!SD.exists(p)) { *out = p; return true; }
  }
  return false;
}

// v002: path of the n-th .csv in a class folder (same order as myCountSamples)
bool myNthSamplePath(int classIdx, int n, String* out) {
  if (!mySDavailable || classIdx < 0 || classIdx >= MY_DATA_FOLDERS || n < 0) return false;
  String path = "/motion/" + myFolderName(classIdx);
  File root = SD.open(path);
  if (!root) return false;
  int count = 0;
  bool found = false;
  while (File f = root.openNextFile()) {
    String name = f.name();
    bool isCsv = !f.isDirectory() && name.endsWith(".csv");
    f.close();
    if (isCsv) {
      if (count == n) { *out = path + "/" + name; found = true; break; }
      count++;
    }
  }
  root.close();
  return found;
}

// v002: read one raw window (g) at SAMPLE_INTERVAL_MS spacing. Raw = not normalized.
void myCaptureWindow(float* w, bool echo) {
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    unsigned long tStart = millis();
    myReadAccel(w + t * IMU_AXES);
    // Brief echo every 10 samples
    if (echo && t % 10 == 0) Serial.printf("  t%02d: %.3f,%.3f,%.3f\n", t, w[t*IMU_AXES], w[t*IMU_AXES+1], w[t*IMU_AXES+2]);
    // Pace to SAMPLE_INTERVAL_MS
    long elapsed = millis() - tStart;
    if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
  }
}

// v002: one csv file, one row per timestep: ax,ay,az  (same format as v001)
bool myWriteSample(int classIdx, const float* w, String* pathOut) {
  String filePath;
  if (!myNewSamplePath(classIdx, &filePath)) return false;
  File f = SD.open(filePath, FILE_WRITE);
  if (!f) { Serial.println("ERROR: cannot open file for writing"); return false; }
  for (int t = 0; t < IMU_TIMESTEPS; t++)
    f.printf("%.5f,%.5f,%.5f\n", w[t*IMU_AXES], w[t*IMU_AXES+1], w[t*IMU_AXES+2]);
  f.close();
  Serial.printf("Saved: %s\n", filePath.c_str());
  if (pathOut) *pathOut = filePath;
  return true;
}

// Capture one IMU window: 40 samples at ~25 ms intervals, save to SD
bool myCaptureSample(int classIdx) {
  float w[INPUT_SIZE];
  Serial.printf("Capturing %d samples\n", IMU_TIMESTEPS);
  myCaptureWindow(w, true);
  return myWriteSample(classIdx, w, nullptr);
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
    myPumpIncoming();                                  // v002: keep the page link alive
    // Serial input
    if (myKeyAvailable()) {
      char c = myKeyRead();
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
// Input layout: [t0_ax, t0_ay, t0_az, t1_ax, ...]  (IMU_TIMESTEPS x IMU_AXES)
// Weight layout: [k x in_axis x out_filter]  index = (k*IMU_AXES + a)*CONV1_FILTERS + f
// Output layout: [step x filter]  index = step*CONV1_FILTERS + f
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

// Full forward pass: Conv1D -> Pool -> Dense1 -> Dense2 -> Output(softmax)
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

// v002: raw csv -> buf (INPUT_SIZE floats, NOT normalized). The page's GET uses this.
bool myReadSampleCsv(const char* path, float* buf) {
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
  return true;
}

// Load one .csv sample from SD into buf (INPUT_SIZE floats) then normalize
bool myLoadSampleFromFile(const char* path, float* buf) {
  if (!myReadSampleCsv(path, buf)) return false;
  myNormalizeInput(buf);
  return true;
}


// v002: the training body, shared by the menu (myActionTrain) and the page ("TRAIN").
// Returns true if it trained. Sends "EP ..." lines to the page after each epoch.
bool myTrainCore() {
  if (!mySDavailable) {
    Serial.println("No SD - cannot train");
    myReply("ERR no SD card - device training needs the SD samples");
    return false;
  }

  // Build training list (only the two trained classes; the test folder is never read here)
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
  for (int c = 0; c < NUM_CLASSES; c++) {                       // v002: both classes are needed for the ML score
    if (classCounts[c] == 0) {
      Serial.printf("No samples for class '%s' - collect both classes before training.\n", myClassLabels[c].c_str());
      myReply("ERR no samples for %s - the ML score needs both %s and %s",
              myClassLabels[c].c_str(), myClassLabels[0].c_str(), myClassLabels[1].c_str());
      return false;
    }
  }

  // Shuffle and split validation
  std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
  int valCount = 0;
  std::vector<TrainingItem> myValData;
  if (VALIDATION_SAMPLES > 0) {
    // Hold out up to VALIDATION_SAMPLES per class, but always leave one training sample per class
    int heldOut[NUM_CLASSES] = {};
    std::vector<TrainingItem> trainOnly;
    for (auto& item : myTrainingData) {
      int cap = classCounts[item.label] - 1;                    // v002
      if (cap > VALIDATION_SAMPLES) cap = VALIDATION_SAMPLES;
      if (heldOut[item.label] < cap) {
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

  // Training-only buffers (grad/Adam) live only inside this function
  if (!myAllocateTrainingMemory()) {
    Serial.println("FATAL: Could not allocate training memory - see allocation errors above.");
    myFreeTrainingMemory();
    myReply("ERR training memory allocation failed");
    return false;
  }
  float* myBatchBuf = (float*)ps_malloc(INPUT_SIZE * sizeof(float));
  if (!myCheckAlloc(myBatchBuf, "myBatchBuf", INPUT_SIZE * sizeof(float))) {
    myFreeTrainingMemory();
    myReply("ERR malloc");
    return false;
  }

  myStopRequested = false;
  int epochsDone = 0;
  for (int epoch = 0; epoch < TARGET_EPOCHS; epoch++) {
    std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
    float epochLoss = 0;
    int   correct   = 0;
    int   processed = 0;
    myZeroGradients();

    for (int si = 0; si < (int)myTrainingData.size(); si++) {
      myCheckTouchBackground();  // keep touch responsive
      myPumpIncoming();                                              // v002
      if (myKeyAvailable() && myKeyRead() == 'x') myStopRequested = true;   // v002
      if (myStopRequested) break;                                    // v002
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
    if (myStopRequested) { Serial.println("Training stopped"); break; }   // v002

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

    float trainAcc = 100.0f * correct / myMaxI((int)myTrainingData.size(), 1);
    float avgLoss  = epochLoss / myMaxI((int)myTrainingData.size(), 1);
    Serial.printf("Epoch %2d/%d  Loss=%.4f  TrainAcc=%.1f%%  ValAcc=%.1f%%\n",
                  epoch + 1, TARGET_EPOCHS, avgLoss, trainAcc, valAcc);
    myReply("EP %d %d %.4f %.1f %.1f", epoch + 1, TARGET_EPOCHS, avgLoss, trainAcc, valAcc);   // v002
    epochsDone++;

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
  myFreeTrainingMemory();   // release grad/Adam buffers - not needed again until the next Train
  myStopRequested = false;
  myWeightsTrained = true;
  mySaveWeights();
  Serial.println("ML weights saved.");

  // Also (re)compute the per-axis statistical baseline from the "1normal" samples:
  // this drives the RMS/P2P/ZCR readout at Infer, independent of the ML model above.
  bool baseOk = myComputeFeatureBaseline();
  myReply("DONE train %d epochs, weights saved=%d, baseline n=%d", epochsDone, mySDavailable ? 1 : 0, baseOk ? myFeatureBaselineCount : 0);   // v002
  return true;
}

void myActionTrain() {
  bool ok = myTrainCore();                                           // v002: body moved to myTrainCore()
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 15, ok ? "Training done!" : "Not trained");
    u8g2.drawStr(0, 28, ok ? "Weights saved" : "See Serial log");
  } while (u8g2.nextPage());
  delay(2000);
  myResetMenuState();
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE FUNCTIONS                                             ██
// ██                                                                          ██
// ██  Continuously captures 1-second IMU windows and scores them.             ██
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

  Serial.println("\n>>> Anomaly inference mode (tap A0 x3 or 'l' to exit)");
  Serial.printf("ML anomaly = 100%% - P(%s).  Flag threshold=%.0f%%\n",
                myClassLabels[myNormalClassIdx].c_str(), ANOMALY_FLAG_PERCENT);
  if (!myFeatureBaselineReady) {
    Serial.println("NOTE: no feature baseline yet - X/Y/Z readout will show 0%.");
    Serial.println("      Run Train again after collecting '1normal' samples to build it.");
  }

  float myLiveBuf[INPUT_SIZE];
  float myRes[MY_RES_N];
  int   windowCount = 0;

  while (true) {
    myPumpIncoming();                                  // v002
    if (myCheckTouchInput() == 2) { myResetMenuState(); return; }
    if (myKeyAvailable()) { char c = myKeyRead(); if (c == 'l' || c == 'L') { myResetMenuState(); return; } }

    // Capture one 1-second window, then score it
    myCaptureWindow(myLiveBuf, false);
    myAnalyzeWindow(myLiveBuf, myRes);
    windowCount++;
    bool isAnomaly = myRes[MY_R_FLAG] > 0.5f;

    Serial.printf("Win %d | ML anomaly=%.0f%% (P(%s)=%.0f%%, raw class=%s) | X=%.0f%% Y=%.0f%% Z=%.0f%% | [%s]\n",
                  windowCount, myRes[MY_R_ML], myClassLabels[myNormalClassIdx].c_str(), myRes[MY_R_PNORM] * 100,
                  myClassLabels[(int)myRes[MY_R_PRED]].c_str(), myRes[MY_R_X], myRes[MY_R_Y], myRes[MY_R_Z],
                  isAnomaly ? "ANOMALY" : "normal");

    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      char buf[20];
      snprintf(buf, sizeof(buf), "ML:%3.0f%% #%d", myRes[MY_R_ML], windowCount);
      u8g2.drawStr(0, 8, buf);
      snprintf(buf, sizeof(buf), "X%3.0f Y%3.0f Z%3.0f", myRes[MY_R_X], myRes[MY_R_Y], myRes[MY_R_Z]);
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
  myBusy = false;                                      // v002
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
  myBusy = true;                                       // v002: page commands answer "BUSY" meanwhile
  if      (idx <= NUM_CLASSES)        myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES + 1)    myActionTrain();
  else                                myActionInfer();
  myBusy = false;
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  if (!myIsSelected && myKeyAvailable()) {             // v002: was Serial.available()
    char c = myKeyRead();
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



// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 5 (v002): LINK TO THE WEB PAGE  (WebBLE + Web Serial)              ██
// ██                                                                          ██
// ██  Same frames on both transports. Frames from BLE arrive in a callback    ██
// ██  and are only QUEUED there; all work happens in loop() / myPumpIncoming. ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// ==LINK START==
// Standard CRC-32 (the same one zip and the page use)
uint32_t myCrc32(const uint8_t* d, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; i++) {
    c ^= d[i];
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
  }
  return ~c;
}

void myPut32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)((v >> 8) & 0xFF);
  p[2] = (uint8_t)((v >> 16) & 0xFF); p[3] = (uint8_t)((v >> 24) & 0xFF);
}

uint32_t myGet32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Send one frame on the transport the last command came in on.
bool mySendFrame(const uint8_t* d, size_t n) {
  if (myReplyVia == 1) {
#if MY_USE_BLE
    if (!myBleConnected || !myEvtChar) return false;
    myEvtChar->setValue(d, n);
    unsigned long t0 = millis();
    bool ok = myEvtChar->notify();
    while (!ok && myBleConnected && millis() - t0 < 1000) { delay(10); ok = myEvtChar->notify(); }
    return ok;
#else
    return false;
#endif
  }
  if (myReplyVia == 2) {
    char line[MY_FRAME_MAX * 2 + 16];
    size_t ol = 0;
    memcpy(line, "@B ", 3);
    if (mbedtls_base64_encode((unsigned char*)line + 3, sizeof(line) - 5, &ol, d, n) != 0) return false;
    line[3 + ol] = '\n';
    Serial.write((const uint8_t*)line, 3 + ol + 1);   // one write so other prints do not split the line
    return true;
  }
  return false;
}

// Text reply line, e.g. myReply("OK model saved=%d", 1)
void myReply(const char* fmt, ...) {
  uint8_t f[MY_FRAME_MAX];
  f[0] = MY_F_RESP;
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf((char*)f + 1, MY_FRAME_MAX - 1, fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if (n > MY_FRAME_MAX - 2) n = MY_FRAME_MAX - 2;
  mySendFrame(f, 1 + n);
}

// next = the next chunk number I expect; status 0 ok, 1 crc error, 2 abort
void myAck(uint16_t next, uint8_t status) {
  uint8_t f[4] = { MY_F_ACK, (uint8_t)(next & 0xFF), (uint8_t)(next >> 8), status };
  mySendFrame(f, 4);
}

// Send a blob (kind 'M' model, 'S' sample window) with acks every MY_WIN chunks.
// Returns false if the page stopped answering or refused it.
bool mySendBlob(uint8_t kind, uint8_t id, const uint8_t* data, uint32_t total) {
  mySending = true;
  uint32_t crc = myCrc32(data, total);
  uint16_t nCh = (uint16_t)((total + MY_CHUNK_DATA - 1) / MY_CHUNK_DATA);
  uint8_t f[MY_FRAME_MAX];
  f[0] = MY_F_HEAD; f[1] = kind; f[2] = id;
  myPut32(f + 3, total); myPut32(f + 7, crc);
  mySendFrame(f, 11);

  uint16_t acked = 0;
  int tries = 0;
  while (acked < nCh) {
    uint16_t end = (uint16_t)(acked + MY_WIN);
    if (end > nCh) end = nCh;
    myTxAckNext = 0xFFFF;
    myTxAckStatus = 0;
    for (uint16_t s = acked; s < end; s++) {
      uint32_t off = (uint32_t)s * MY_CHUNK_DATA;
      uint32_t n = total - off;
      if (n > MY_CHUNK_DATA) n = MY_CHUNK_DATA;
      f[0] = MY_F_DATA; f[1] = (uint8_t)(s & 0xFF); f[2] = (uint8_t)(s >> 8);
      memcpy(f + 3, data + off, n);
      mySendFrame(f, 3 + n);
      delay(8);
    }
    unsigned long t0 = millis();
    while (myTxAckNext == 0xFFFF && millis() - t0 < MY_ACK_TIMEOUT_MS) { myPumpIncoming(); delay(2); }
    if (myTxAckNext == 0xFFFF) {                       // no answer: resend this window
      if (++tries > 10) { mySending = false; return false; }
      if (acked == 0) {                                // the page may have missed the header
        f[0] = MY_F_HEAD; f[1] = kind; f[2] = id; myPut32(f + 3, total); myPut32(f + 7, crc);
        mySendFrame(f, 11);
      }
      continue;
    }
    if (myTxAckStatus == 2) { mySending = false; return false; }   // the page aborted
    if (myTxAckNext > acked) { acked = myTxAckNext; tries = 0; }
    else {                                             // "still expecting chunk N": something was lost
      if (++tries > 10) { mySending = false; return false; }
      unsigned long t1 = millis();                     // the receiver sends one such nack per later chunk:
      while (millis() - t1 < 80) { myPumpIncoming(); delay(2); }   // let this round's duplicates drain, then resend
    }
  }
  mySending = false;
  return true;
}

// ---- receiving a blob from the page ----
void myOnHead(const uint8_t* d) {
  uint8_t  kind  = d[1];
  uint8_t  id    = d[2];
  uint32_t total = myGet32(d + 3);
  uint32_t crc   = myGet32(d + 7);
  if (myBusy) { myReply("ERR busy - try again when the device is idle"); myAck(0, 2); return; }
  bool ok = false;
  if (kind == 'W')      ok = (total == MY_PACKAGE_BYTES);
  else if (kind == 'S') ok = (total == INPUT_SIZE * 4 && id < MY_DATA_FOLDERS);
  else if (kind == 'C') ok = (total > 0 && total <= 4096);
  if (!ok) {
    myReply("ERR refused blob %c id=%u size=%lu - this sketch needs W=%u S=%u (folders %u)",
            (char)kind, (unsigned)id, (unsigned long)total,
            (unsigned)MY_PACKAGE_BYTES, (unsigned)(INPUT_SIZE * 4), (unsigned)MY_DATA_FOLDERS);
    myAck(0, 2);
    return;
  }
  myRxKind = kind; myRxId = id; myRxTotal = total; myRxCrc = crc;
  myRxGot = 0; myRxNext = 0; myRxLastMs = millis();
  myRxActive = true;
  myAck(0, 0);
}

void myOnData(const uint8_t* d, size_t n) {
  if (!myRxActive) return;
  uint16_t seq = (uint16_t)(d[1] | (d[2] << 8));
  myRxLastMs = millis();
  if (seq != myRxNext) { myAck(myRxNext, 0); return; }      // out of order: say what I expect
  uint32_t off = (uint32_t)seq * MY_CHUNK_DATA;
  uint32_t len = (uint32_t)(n - 3);
  if (off + len > myRxTotal) { myRxActive = false; myAck(myRxNext, 2); return; }
  memcpy(myBlobBuf + off, d + 3, len);
  myRxNext++;
  myRxGot += len;
  if (myRxGot >= myRxTotal) {
    myRxActive = false;
    if (myCrc32(myBlobBuf, myRxTotal) == myRxCrc) {
      myAck(myRxNext, 0);
      myDispatchBlob(myRxKind, myRxId, myRxTotal);
    } else {
      myAck(myRxNext, 1);
      myReply("ERR crc mismatch - transfer discarded");
    }
  } else if ((myRxNext % MY_WIN) == 0) {
    myAck(myRxNext, 0);
  }
}

void myHandleFrame(const uint8_t* d, size_t n) {
  if (n < 1) return;
  switch (d[0]) {
    case MY_F_HEAD: if (n == 11) myOnHead(d); break;
    case MY_F_DATA: if (n > 3)   myOnData(d, n); break;
    case MY_F_ACK:  if (n >= 4) {                      // keep the HIGHEST ack of this round: a late duplicate nack must not undo progress
      uint16_t nx = (uint16_t)(d[1] | (d[2] << 8));
      if (d[3] == 2) { myTxAckStatus = 2; if (myTxAckNext == 0xFFFF) myTxAckNext = nx; }
      else if (myTxAckNext == 0xFFFF || nx >= myTxAckNext) { if (myTxAckStatus != 2 || myTxAckNext == 0xFFFF) myTxAckStatus = d[3]; myTxAckNext = nx; }
    } break;
    case MY_F_TEXT: {
      char s[MY_FRAME_MAX];
      memcpy(s, d + 1, n - 1);
      s[n - 1] = 0;
      myHandleCommand(s);
    } break;
    default: break;
  }
}

// BLE callback -> queue. Called from the BLE task, so it only copies.
void myQPush(const uint8_t* d, size_t n, uint8_t via) {
  if (n == 0 || n > MY_FRAME_MAX) return;
  uint8_t next = (uint8_t)((myQHead + 1) % MY_QN);
  if (next == myQTail) return;                         // full: drop it, the ack protocol resends
  myQ[myQHead].len = (uint8_t)n;
  myQ[myQHead].via = via;
  memcpy(myQ[myQHead].d, d, n);
  myQHead = next;
}

// Web Serial: lines that start with '@' are frames. Anything else is left for the menu.
void myPollSerialFrames() {
  while (Serial.available()) {
    if (mySerLen == 0 && Serial.peek() != '@') return;
    int c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (mySerLen > 0) {
        mySerLine[mySerLen] = 0;
        uint8_t f[MY_FRAME_MAX + 4];
        size_t ol = 0;
        int len = mySerLen;
        mySerLen = 0;                                  // free the line buffer before handling
        if (len > 3 && mySerLine[1] == 'B' && mySerLine[2] == ' ' &&
            mbedtls_base64_decode(f, sizeof(f), &ol, (const unsigned char*)mySerLine + 3, len - 3) == 0 && ol > 0) {
          myReplyVia = 2;
          myHandleFrame(f, ol);
        }
        if (mySending && myTxAckNext != 0xFFFF) return;   // the awaited ack is in: leave the rest for after this command
      }
      continue;
    }
    if (mySerLen < (int)sizeof(mySerLine) - 1) mySerLine[mySerLen++] = (char)c;
    else mySerLen = 0;                                 // too long: drop the line
  }
}

// Handle everything waiting. Safe to call from long loops (training, sending).
void myPumpIncoming() {
  myPollSerialFrames();
  while (myQTail != myQHead) {
    MyFrame fr = myQ[myQTail];                         // copy first, then advance (handlers may re-enter)
    myQTail = (uint8_t)((myQTail + 1) % MY_QN);
    myReplyVia = fr.via;
    myHandleFrame(fr.d, fr.len);
    if (mySending && myTxAckNext != 0xFFFF) break;     // same rule as the serial path
  }
  if (myRxActive && millis() - myRxLastMs > 6000) {
    myRxActive = false;
    myReply("ERR receive timeout - transfer dropped");
  }
}
// ==LINK END==


// ---- BLE server (NimBLE-Arduino 2.x calls, as used in the fusion firmware) ----
#if MY_USE_BLE
class MyBleServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* s, NimBLEConnInfo& info) override {
    myBleConnected = true;
    Serial.println("Browser connected (BLE)");
  }
  void onDisconnect(NimBLEServer* s, NimBLEConnInfo& info, int reason) override {
    myBleConnected = false;
    myRxActive = false;
    Serial.println("Browser disconnected - advertising again");
    NimBLEDevice::startAdvertising();
  }
};

class MyBleCmdCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& info) override {
    std::string v = c->getValue();
    myQPush((const uint8_t*)v.data(), v.size(), 1);    // work happens in loop()
  }
};

void myStartBle() {
  NimBLEDevice::init(MY_DEVICE_NAME);
  NimBLEDevice::setMTU(247);
  myBleServer = NimBLEDevice::createServer();
  myBleServer->setCallbacks(new MyBleServerCallbacks());
  NimBLEService* svc = myBleServer->createService(MY_SVC_UUID);
  myCmdChar = svc->createCharacteristic(MY_CMD_UUID, NIMBLE_PROPERTY::WRITE);
  myEvtChar = svc->createCharacteristic(MY_EVT_UUID, NIMBLE_PROPERTY::NOTIFY);
  myCmdChar->setCallbacks(new MyBleCmdCallbacks());
  svc->start();

  // Advertise the NAME only: a 128-bit UUID + name overflows the 31-byte packet and
  // advertising then fails silently (the page filters by name prefix anyway).
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->setName(MY_DEVICE_NAME);
  if (adv->start()) {
    Serial.print("BLE advertising as \"");
    Serial.print(MY_DEVICE_NAME);
    Serial.println("\" - open index-anomaly-v002.html and press Connect BLE");
  } else {
    Serial.println("ERROR: BLE advertising failed to start - the page will not see this board");
  }
}
#endif


// ---- what the page can ask for ----
// Text commands:  STATUS  DBG 1|0  STOP  CAPTURE c  GET c n  DEL c n  TRAIN  INFER  GETMODEL  CALIB  BASELINE
//   (c = 0 off, 1 normal, 2 test anomaly; the "LST" line of STATUS lists the three counts)
// Blobs from the page: 'W' model package, 'S' sample window for folder id, 'C' config.json text.
void myReplyInfo() {
  myReply("INFO T=%d A=%d K=%d F=%d D1=%d D2=%d C=%d INT=%d W=%d SD=%d TR=%d FB=%d",
          IMU_TIMESTEPS, IMU_AXES, CONV1_KERNEL, CONV1_FILTERS, DENSE1_SIZE, DENSE2_SIZE,
          NUM_CLASSES, SAMPLE_INTERVAL_MS, MY_WEIGHT_FLOATS, mySDavailable ? 1 : 0, myWeightsTrained ? 1 : 0,
          myFeatureBaselineReady ? myFeatureBaselineCount : 0);
  char buf[MY_FRAME_MAX];
  int n = snprintf(buf, sizeof(buf), "LBL ");
  for (int i = 0; i < NUM_CLASSES && n < (int)sizeof(buf) - 2; i++)
    n += snprintf(buf + n, sizeof(buf) - n, "%s%s", i ? "," : "", myClassLabels[i].c_str());
  myReply("%s", buf);
  n = snprintf(buf, sizeof(buf), "LST");
  for (int i = 0; i < MY_DATA_FOLDERS && n < (int)sizeof(buf) - 8; i++)
    n += snprintf(buf + n, sizeof(buf) - n, " %d", myCountSamples(i));
  myReply("%s", buf);
  myReply("ANOM %.2f %.3f", ANOMALY_FLAG_PERCENT, FEATURE_SIGMA);
  myReplyCal();                                        // last line: the page waits for it
}

void myReplyCal() {
  myReply("CAL %.5f %.5f %.5f %.5f %.5f %.5f",
          myAccelMean[0], myAccelMean[1], myAccelMean[2], myAccelStd[0], myAccelStd[1], myAccelStd[2]);
}

void myImportPackage() {
  if (!myPackFromBuf((const float*)myBlobBuf)) { myReply("ERR model holds NaN or Infinity - rejected"); return; }
  myWeightsTrained = true;
  mySaveWeights();
  mySaveCalib();
  if (myFeatureBaselineReady) mySaveFeatureBaseline();
  myReply("OK model loaded, saved to SD=%d baseline n=%d", mySDavailable ? 1 : 0, myFeatureBaselineReady ? myFeatureBaselineCount : 0);
}

void myStoreSample(int classIdx) {
  if (!mySDavailable) { myReply("ERR no SD card - sample not stored"); return; }
  String p;
  if (myWriteSample(classIdx, (const float*)myBlobBuf, &p)) myReply("OK put %d %s", classIdx, p.c_str());
  else myReply("ERR could not write the sample");
}

void myStoreConfig(uint32_t total) {
  myBlobBuf[total] = 0;
  if (mySDavailable) {
    if (!SD.exists("/header")) SD.mkdir("/header");
    File f = SD.open("/header/config.json", FILE_WRITE);
    if (f) { f.write(myBlobBuf, total); f.close(); }
  }
  myApplyConfig((const char*)myBlobBuf);
  myReply("OK config %s", mySDavailable ? "saved" : "applied (no SD)");
}

void myDispatchBlob(uint8_t kind, uint8_t id, uint32_t total) {
  if (kind == 'W')      myImportPackage();
  else if (kind == 'S') myStoreSample(id);
  else if (kind == 'C') myStoreConfig(total);
}

void myHandleCommand(char* s) {
  if (!strncmp(s, "DBG ", 4)) { myLinkDebugOn = (s[4] == '1'); myDebugLastMs = millis(); return; }
  if (!strcmp(s, "STOP"))     { myStopRequested = true; return; }
  if (!strcmp(s, "STATUS"))   { myReplyInfo(); return; }
  if (myBusy || myRxActive)   { myReply("BUSY"); return; }
  myBusy = true;
  if (!strncmp(s, "CAPTURE ", 8)) {
    int c = atoi(s + 8);
    if (c < 0 || c >= MY_DATA_FOLDERS) {
      myReply("ERR folder %d out of range", c);
    } else {
      float w[INPUT_SIZE];
      myCaptureWindow(w, false);
      String p;
      if (mySDavailable && myWriteSample(c, w, &p)) myReply("OK cap %d %s", c, p.c_str());
      else myReply("WARN cap %d not saved (no SD or write failed)", c);
      mySendBlob('S', (uint8_t)c, (const uint8_t*)w, INPUT_SIZE * 4);
    }
  } else if (!strncmp(s, "GET ", 4)) {
    int c = -1, n = -1;
    sscanf(s + 4, "%d %d", &c, &n);
    String p;
    if (myNthSamplePath(c, n, &p) && myReadSampleCsv(p.c_str(), (float*)myBlobBuf))
      mySendBlob('S', (uint8_t)c, myBlobBuf, INPUT_SIZE * 4);
    else myReply("ERR get %d %d", c, n);
  } else if (!strncmp(s, "DEL ", 4)) {
    int c = -1, n = -1;
    sscanf(s + 4, "%d %d", &c, &n);
    String p;
    if (myNthSamplePath(c, n, &p) && SD.remove(p)) myReply("OK del %d %d", c, n);
    else myReply("ERR del %d %d", c, n);
  } else if (!strcmp(s, "TRAIN")) {
    myTrainCore();
  } else if (!strcmp(s, "CALIB")) {
    myCalibrate(true);
    myReplyCal();
  } else if (!strcmp(s, "BASELINE")) {
    if (myComputeFeatureBaseline()) myReply("OK baseline n=%d", myFeatureBaselineCount);
    else myReply("ERR baseline needs samples in %s on the SD card", myClassLabels[myNormalClassIdx].c_str());
  } else if (!strcmp(s, "GETMODEL")) {
    if (!myWeightsTrained) {
      myReply("ERR no trained model on the device yet");
    } else {
      myPackToBuf((float*)myBlobBuf);
      mySendBlob('M', 0, myBlobBuf, MY_PACKAGE_BYTES);
    }
  } else if (!strcmp(s, "INFER")) {
    if (!myWeightsTrained) {
      myReply("ERR no trained model on the device yet");
    } else {
      float w[INPUT_SIZE];
      float r[MY_RES_N];
      myCaptureWindow(w, false);
      myAnalyzeWindow(w, r);
      myReply("RES %.3f %.4f %.2f %.2f %.2f %d %d", r[MY_R_ML], r[MY_R_PNORM], r[MY_R_X], r[MY_R_Y], r[MY_R_Z],
              (int)r[MY_R_FLAG], (int)r[MY_R_PRED]);
      if (myLinkDebugOn) {                              // for the page's parity check: the 9 live features, then the raw window
        char buf[MY_FRAME_MAX];
        int n = snprintf(buf, sizeof(buf), "FT");
        for (int k = 0; k < TOTAL_FEATURES && n < (int)sizeof(buf) - 12; k++)
          n += snprintf(buf + n, sizeof(buf) - n, " %.5f", r[MY_R_FEAT + k]);
        myReply("%s", buf);
        mySendBlob('S', 255, (const uint8_t*)w, INPUT_SIZE * 4);
      }
    }
  } else {
    myReply("ERR unknown command: %s", s);
  }
  myBusy = false;
}

// Live ax,ay,az for the page while "debug frames" is on (the page re-sends DBG 1 every 5 s)
void myLinkHeartbeat() {
  unsigned long now = millis();
  if (myLinkDebugOn && now - myDebugLastMs > 15000) myLinkDebugOn = false;   // page went away
  if (myLinkDebugOn != myLinkWasDebug) {
    myLinkWasDebug = myLinkDebugOn;
    Serial.println(myLinkDebugOn ? "Debug frames ON" : "Debug frames OFF");
  }
  if (!myLinkDebugOn || myBusy || myRxActive || myReplyVia == 0) return;
  if (now - myLastHbMs < 250) return;
  myLastHbMs = now;
  float v[IMU_AXES];
  myReadAccel(v);
  myReply("HB %.3f %.3f %.3f", v[0], v[1], v[2]);
}


// ======================================================
// NOTE ON SENSOR FUSION EXTENSION
// ======================================================
// The 120-input vector is currently 40 x [ax, ay, az].
// To extend to other sensor combinations, change the layout here:
//
//   IMU_AXES = 6 -> [ax, ay, az, gx, gy, gz]   40 x 6 = 240  (update INPUT_SIZE = 240)
//   IMU_AXES = 2 -> [ax, ay]                    60 x 2 = 120  (adjust IMU_TIMESTEPS = 60)
//   Mixed sensors -> concatenate channels in the buffer, one entry per timestep
//
// (v002 keeps IMU_AXES = 3 because the page, the phone mapping and myReadAccel() all assume it.)
// ======================================================
