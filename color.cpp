#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <array>

using namespace cv;
using namespace std;

/* ---------------- CONFIG ---------------- */
const string MODEL_PATH = "model7.onnx";
const string VIDEO_PATH = "istockphoto-1478951186-640_adpp_is.mp4";

const int   INPUT_SIZE = 640;
const float CONF_THRESHOLD = 0.25f;
const float IOU_THRESHOLD  = 0.45f;
const int   TARGET_CLASS_ID = 2;

/* ---------------- HSV COLOR CLASSIFIER ---------------- */

string classify_color_hsv(Mat &frame, int x1, int y1, int x2, int y2)
{
    x1 = max(0, x1); y1 = max(0, y1);
    x2 = min(frame.cols - 1, x2);
    y2 = min(frame.rows - 1, y2);

    if (x2 <= x1 || y2 <= y1) return "OTHER";

    Mat roi = frame(Rect(x1, y1, x2 - x1, y2 - y1));
    Mat hsv;
    cvtColor(roi, hsv, COLOR_BGR2HSV);

    vector<int> H, S, V;
    H.reserve(hsv.total());
    S.reserve(hsv.total());
    V.reserve(hsv.total());

    for (int y = 0; y < hsv.rows; y++)
        for (int x = 0; x < hsv.cols; x++) {
            Vec3b p = hsv.at<Vec3b>(y, x);
            H.push_back(p[0]);
            S.push_back(p[1]);
            V.push_back(p[2]);
        }

    auto median = [](vector<int> &v) {
        nth_element(v.begin(), v.begin() + v.size()/2, v.end());
        return v[v.size()/2];
    };

    int h = median(H);
    int s = median(S);
    int v = median(V);

    // Ignore dull pixels
    if (s < 60 || v < 60) return "OTHER";

    // RED/ORANGE: 0–40 (orange) + 160–179 (red)
    if ((h >= 0 && h <= 40) || (h >= 160 && h <= 179)) return "RED";

    // GREEN: 41–85
    if (h >= 41 && h <= 85) return "GREEN";

    // BLUE: 86–130
    if (h >= 86 && h <= 130) return "BLUE";

    return "OTHER";
}

/* ---------------- MAIN ---------------- */

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, MODEL_PATH.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name  = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    const char* input_names[]  = { input_name.get() };
    const char* output_names[] = { output_name.get() };

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        cerr << "ERROR opening video\n";
        return -1;
    }

    Mat frame;

    while (cap.read(frame)) {

        int H = frame.rows;
        int W = frame.cols;

        Mat blob;
        dnn::blobFromImage(
            frame, blob,
            1.0 / 255.0,
            Size(INPUT_SIZE, INPUT_SIZE),
            Scalar(), true, false
        );

        array<int64_t, 4> input_shape{1, 3, INPUT_SIZE, INPUT_SIZE};

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                mem_info,
                (float*)blob.ptr<float>(),
                blob.total(),
                input_shape.data(),
                input_shape.size()
            );

        auto t0 = chrono::high_resolution_clock::now();

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        auto t1 = chrono::high_resolution_clock::now();
        float fps = 1.0f /
            chrono::duration<float>(t1 - t0).count();

        float* out = outputs[0].GetTensorMutableData<float>();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        int num_channels = shape[1];
        int num_preds    = shape[2];
        int num_classes  = num_channels - 4;

        vector<Rect> boxes;
        vector<float> scores;

        for (int i = 0; i < num_preds; i++) {

            float cx = out[0 * num_preds + i];
            float cy = out[1 * num_preds + i];
            float w  = out[2 * num_preds + i];
            float h  = out[3 * num_preds + i];

            float best = 0.f;
            int best_id = -1;

            for (int c = 0; c < num_classes; c++) {
                float cls = out[(4 + c) * num_preds + i];
                if (cls > best) {
                    best = cls;
                    best_id = c;
                }
            }

            if (best_id != TARGET_CLASS_ID || best < CONF_THRESHOLD)
                continue;

            int x1 = int((cx - w / 2) * W / INPUT_SIZE);
            int y1 = int((cy - h / 2) * H / INPUT_SIZE);
            int bw = int(w * W / INPUT_SIZE);
            int bh = int(h * H / INPUT_SIZE);

            boxes.emplace_back(x1, y1, bw, bh);
            scores.push_back(best);
        }

        vector<int> keep;
        dnn::NMSBoxes(boxes, scores,
                      CONF_THRESHOLD,
                      IOU_THRESHOLD,
                      keep);

        for (int i : keep) {
            Rect r = boxes[i];
            string color = classify_color_hsv(
                frame, r.x, r.y,
                r.x + r.width,
                r.y + r.height
            );

            Scalar col(255,255,255);
            if (color == "RED") col = Scalar(0,165,255);
            if (color == "GREEN") col = Scalar(0,255,0);
            if (color == "BLUE") col = Scalar(255,0,0);

            rectangle(frame, r, col, 2);
            putText(frame, color, r.tl(),
                    FONT_HERSHEY_SIMPLEX, 0.6, col, 2);
        }

        putText(frame, "FPS: " + to_string((int)fps),
                Point(10,30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 2);

        imshow("YOLOv8 Color", frame);
        if (waitKey(1) == 'q') break;
    }

    return 0;
}

