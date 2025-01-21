#include <assert.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
class U2NetModel
{
public:
    U2NetModel(const wchar_t* onnx_model_path);
    std::vector<float> predict(std::vector<float>& input_data,int batch_size=1,int index=0);
    cv::Mat predict(cv::Mat& input_tensor, int batch_size = 1, int index = 0);
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*>input_node_names;
    std::vector<const char*>output_node_names;
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
};
U2NetModel::U2NetModel(const wchar_t* onnx_model_path):session(nullptr),env(nullptr)
{
    //初始化环境，每个进程一个环境,环境保留了线程池和其他状态信息
    this->env=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "u2net");
    //初始化Session选项
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 创建Session并把模型加载到内存中
    this->session=Ort::Session(env, onnx_model_path,session_options);
    //输入输出节点数量和名称
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    for (int i = 0; i < num_input_nodes; i++)
    {
        auto input_node_name = session.GetInputName(i, allocator);
        this->input_node_names.push_back(input_node_name);
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        this->input_node_dims = tensor_info.GetShape();
    }
    for (int i = 0; i < num_output_nodes; i++)
    {
        auto output_node_name = session.GetOutputName(i, allocator);
        this->output_node_names.push_back(output_node_name);
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        this->output_node_dims = tensor_info.GetShape();
    }
}
std::vector<float> U2NetModel::predict(std::vector<float>& input_tensor_values,int batch_size,int index)
{
    this->input_node_dims[0] = batch_size;
    this->output_node_dims[0] = batch_size;
    float* floatarr = nullptr;
    try
    {
        std::vector<const char*>output_node_names;
        if (index != -1)
        {
            output_node_names = { this->output_node_names[index] };
        }
        else
        {
            output_node_names = this->output_node_names;
        }
        this->input_node_dims[0] = batch_size;
        auto input_tensor_size = input_tensor_values.size();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        floatarr = output_tensors.front().GetTensorMutableData<float>();
    }
    catch (Ort::Exception&e)
    {
        throw e;
    }
    int64_t output_tensor_size = 1;
    for (auto& it : this->output_node_dims)
    {
        output_tensor_size *= it;
    }
    std::vector<float>results(output_tensor_size);
    for (unsigned i = 0;i < output_tensor_size; i++)
    {
        results[i] = floatarr[i];
    }
    return results;
}
cv::Mat U2NetModel::predict(cv::Mat& input_tensor, int batch_size, int index)
{
    int input_tensor_size = input_tensor.cols * input_tensor.rows * 3;
    std::size_t counter = 0;//std::vector空间一次性分配完成，避免过多的数据copy
    std::vector<float>input_data(input_tensor_size);
    std::vector<float>output_data;
    try
    {
        for (unsigned k = 0; k < 3; k++)
        {
            for (unsigned i = 0; i < input_tensor.rows; i++)
            {
                for (unsigned j = 0; j < input_tensor.cols; j++)
                {
                    input_data[counter++]=static_cast<float>(input_tensor.at<cv::Vec3b>(i, j)[k]) / 255.0;
                }
            }
        }
    }
    catch (cv::Exception& e)
    {
        printf(e.what());
    }
    try
    {
        output_data = this->predict(input_data);
    }
    catch (Ort::Exception& e)
    {
        throw e;
    }
    cv::Mat output_tensor(output_data);
    output_tensor=255.0-output_tensor.reshape(1, { 960,640 })*255.0;
    cv::threshold(output_tensor, output_tensor, 220, 255, cv::THRESH_BINARY_INV);
    
    return output_tensor;
}
int main(int argc, char* argv[]) 
{
    U2NetModel model("/home/kingargroo/cpp/unet/bestr50models.onnx");
    cv::Mat image = cv::imread("/home/kingargroo/cpp/unet/test.jpg");
    cv::resize(image, image, { 960,640 },0.0,0.0, cv::INTER_CUBIC);//调整大小到320*320
    cv::imshow("image", image);                                     //打印原图片
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);                  //BRG格式转化为RGB格式
    auto result=model.predict(image);                               //模型预测
    cv::imshow("result", result);                                   //打印结果
    cv::waitKey(0);
}
