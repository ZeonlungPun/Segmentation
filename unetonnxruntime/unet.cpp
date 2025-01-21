#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <onnxruntime_cxx_api.h>


class unetInference
{
public:
    Ort::Env* env;
    Ort::Session* session;
    std::string onnx_path_name,result_save_path;
    int model_input_w, model_input_h, model_output_h, model_output_w;
    int neww;
    int newh;
    cv::Mat ori_image;
    int ori_h;
    int ori_w;
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    int num_class;

    unetInference( 
    std::string onnx_path_name,
    int model_input_h,int model_input_w,
    cv::Mat ori_image,std::string result_save_path):
    onnx_path_name(onnx_path_name),result_save_path(result_save_path),
    ori_image(ori_image),
    session(nullptr),env(nullptr)
    {
        this->model_input_w = 0;
        this->model_input_h = 0;
        this->model_output_h = 0;
        this->model_output_w = 0;
        this->ori_h=ori_image.rows;
        this->ori_w=ori_image.cols;
        this->newh=0;
        this->neww=0;
        this->num_class=0;
        //only create model once in heap avoiding memory leap
        this->initialize_model();
        // read the model and get basic parameters
        this->read_model();
    }

    // Lazy initialization of session options
    Ort::SessionOptions& get_session_options() {
        static Ort::SessionOptions session_options;
        static bool is_initialized = false;

        if (!is_initialized) {
            session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
            is_initialized = true;
        }

        return session_options;
    }

    void initialize_model() {
        try {
            if (!this->env) {
                this->env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "unet-onnx");
            }
            // Use the lazy initialized session_options
            Ort::SessionOptions& session_options = get_session_options();  

            if (!this->session) {
                //std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
                this->session = new Ort::Session(*this->env, this->onnx_path_name.c_str(), session_options);
                std::cout << "Model session created successfully." << std::endl;
            }


        } catch (const Ort::Exception& e) {
            std::cerr << "Error during ONNX Runtime initialization: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error occurred during initialization." << std::endl;
        }
    }

    cv::Mat PreprocessImage()
    {
        //letter-box image preprocess
        //original image size
        float iw=this->ori_image.cols;
        float ih=this->ori_image.rows;
        float scale= std::min( this->model_input_w/iw,this->model_input_h/ih);
        // eqaul ratio resize
        neww =static_cast<int>(iw*scale);
        newh =static_cast<int>(ih*scale);
        cv::resize(this->ori_image,this->ori_image,cv::Size(this->neww,this->newh),cv::INTER_LINEAR);
        // target image size
        cv::Size model_inputsize(this->model_input_w,this->model_input_h);
        cv::Mat new_image(model_inputsize,CV_8UC3,cv::Scalar(128,128,128));
        // Paste the image into the new image at the calculated position
        this->ori_image.copyTo(new_image(cv::Rect((this->model_input_w - this->neww) / 2, (this->model_input_h - this->newh) / 2, this->neww, this->newh)));


        return new_image;
    }

    std::vector<float> softmax(std::vector<float> input)
    {
        float total=0;
        float MAX=input[0];
        for(auto x:input)
        {
            MAX=std::max(x,MAX);
        }
        //分子分母同時減去一個最大值，防止數值爆表
        for(auto x:input)
        {
            total+=std::exp(x-MAX);
        }
        std::vector<float> result;
        for(auto x:input)
        {
            result.push_back(std::exp(x-MAX)/total);
        }
        return result;
    }

    void read_model()
    {
        // 獲取模型輸入和輸出信息   
        size_t numInputNodes = this->session->GetInputCount();
        size_t numOutputNodes = this->session->GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        this->input_node_names.reserve(numInputNodes);

        // 解析模型輸入信息
        for (int i = 0; i < numInputNodes; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            this->model_input_w = input_dims[3];
            this->model_input_h = input_dims[2];
            std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
        }

        // 解析模型輸出信息
        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        this->num_class = output_dims[1];
        this->model_output_h =output_dims[2];
        this->model_output_w =output_dims[3];

        std::cout << "output format : Nx dim = " << output_dims[0] << "x" << this->num_class <<"x"<<this->model_output_h<<"x"<<this->model_output_w <<std::endl;
        for (int i = 0; i < numOutputNodes; i++) {
            auto out_name = session->GetOutputNameAllocated(i, allocator);
            this->output_node_names.push_back(out_name.get());
        }
    }


    cv::Mat UNetInference() {
    
        
        // 處理輸入圖像: letter-box
        cv::Mat image=PreprocessImage();

        // 創建 blob: resize and normalize
        cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->model_input_w, this->model_input_h), cv::Scalar(0, 0, 0), true, false);

        // 確認 blob 的尺寸
        std::cout << "Blob shape: " << blob.size() << std::endl;

        size_t tpixels = this->model_input_h * this->model_input_w * 3;
        std::array<int64_t, 4> input_shape_info{ 1, 3 ,this->model_input_h, this->model_input_w };
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

        const std::array<const char*, 1> inputNames = { this->input_node_names[0].c_str() };
        const std::array<const char*, 1> outNames = { this->output_node_names[0].c_str() };

        // 進行推理
        std::vector<Ort::Value> ort_outputs;
        try {
            ort_outputs = session->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
        }
        catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }

        input_tensor_.release();

        // 解析輸出結果
        const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

    
        cv::Mat prediction_map(this->model_output_h,this->model_output_w,CV_32F);
        
        for (int y = 0; y < this->model_output_h; ++y) {
            for (int x = 0; x < this->model_output_w; ++x) {

                std::vector<float> temp_array;
                for (int c = 0; c < this->num_class; ++c) {
                    float prob = pdata[c * this->model_output_h * this->model_output_w + y * this->model_output_w + x]; 
                    temp_array.push_back(prob);
                }
                std::vector<float>softmax_array=softmax(temp_array);
                auto max_iter = std::max_element(softmax_array.begin(), softmax_array.end());
                int  predicted_class=std::distance(softmax_array.begin(), max_iter);
                prediction_map.at<int>(y, x) = predicted_class;
            }
        }


        
        return prediction_map;

    }

    cv::Mat PostProcess(cv::Mat prediction_map) {
        // 去掉 letterbox 操作產生的灰條 remove gray area produced by letter-box
        int start_row = (this->model_input_h - this->newh) / 2;
        int start_col = (this->model_input_w - this->neww) / 2;
        int end_row = start_row + this->newh;
        int end_col = start_col + this->neww;

        // 檢查切割範圍是否超出圖像邊界
        if(start_row < 0 || start_col < 0 || end_row > prediction_map.rows || end_col > prediction_map.cols) {
            std::cerr << "Error: Slicing range is out of bounds." << std::endl;
            return cv::Mat();
        }
        prediction_map = prediction_map(cv::Range(start_row, end_row), cv::Range(start_col, end_col));
        // resize to original size (or interpolation)
        // cv::resize 只能輸入CV_32F或者CV_8U
        cv::resize(prediction_map, prediction_map, cv::Size(this->ori_w, this->ori_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat pred_class =prediction_map;
        // 使用 map 來統計 pred_class 中每個類別的數量
        // do some statistic
        std::map<int, int> class_count;

        for (int i = 0; i < pred_class.rows; ++i) {
            for (int j = 0; j < pred_class.cols; ++j) {
                int class_id = pred_class.at<int>(i, j);
                class_count[class_id]++;
            }
        }
        for (const auto& entry : class_count) {
            std::cout << "Class " << entry.first << " appears " << entry.second << " times." << std::endl;
        }

        return pred_class;
    }

    void Visualize(cv::Mat pred_class) {
        // 預先設置顏色映射表（每個類別對應一個顏色）
        std::vector<cv::Vec3b> class_colors(this->num_class);

        // 為每個類別生成一個顏色，這裡可以使用固定的顏色
        // 這樣每個類別有相同的顏色，每次顯示的顏色都一致
        class_colors[0] = cv::Vec3b(0, 0, 0);
        for (int i = 1; i < this->num_class; i++) {
            int r = std::rand() % 256;
            int g = std::rand() % 256;
            int b = std::rand() % 256;
            class_colors[i] = cv::Vec3b(b, g, r);  // OpenCV 使用 BGR 順序
        }

        // 創建一個與 pred_class 大小相同的圖像，用來顯示顏色映射
        cv::Mat segmented_image = cv::Mat::zeros(pred_class.size(), CV_8UC3);

        // 遍歷每個像素，將類別索引對應到顏色
        for (int y = 0; y < pred_class.rows; ++y) {
            for (int x = 0; x < pred_class.cols; ++x) {
                int class_id = pred_class.at<int>(y, x);  // 獲取該像素的預測類別索引
                if (class_id > 0 && class_id < this->num_class) {
                    segmented_image.at<cv::Vec3b>(y, x) = class_colors[class_id];  // 根據類別設置顏色
                }
            }
        }

     
        cv::imwrite(this->result_save_path, segmented_image);
        
    }

    void main_process()
    {
        
        cv::Mat prediction_map =UNetInference();
        cv::Mat pred_class= PostProcess(prediction_map);
        Visualize(pred_class);

    }

};





int main()
{
    cv::Mat img=cv::imread("/home/kingargroo/cpp/unet/test.jpg");
    std::string model_path="/home/kingargroo/cpp/unet/bestr50models.onnx";
    std::string result_save_path="/home/kingargroo/cpp/unet/result.png";
    unetInference inference(model_path,640,960,img,result_save_path);
    inference.main_process();
    
    return 0;
    

    
}
