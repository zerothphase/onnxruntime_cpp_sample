#include <iostream>
#include <string>
#include <vector>
#include <random>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

std::default_random_engine generator{ 50 };
std::uniform_real_distribution<float> distribution{ 0.0, 0.1 };

class ONNX_Model
{
public:
    static constexpr const int width_  = 28;
    static constexpr const int height_ = 28;

    std::array<float, width_* height_> input_image_{};
    std::array<float, 10>              results_{};
    int64_t                            result_{ 0 };

    ONNX_Model() 
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_    = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
        output_tensor_   = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run() 
    {
        const char* input_names[] = { "Input3" };
        const char* output_names[] = { "Plus214_Output_0" };

        session_.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        return result_;
    }

    std::vector<int64_t> get_input_shape_from_session()
    {
        Ort::TypeInfo info = session_.GetInputTypeInfo(0);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        size_t dim_count = tensor_info.GetDimensionsCount();
        std::vector<int64_t> dims(dim_count);
        tensor_info.GetDimensions(dims.data(), dims.size());
        return dims;
    }


private:
    //Ort::Env env;
    Ort::Env                env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::Session            session_{ env, L"model.onnx", Ort::SessionOptions{nullptr} };

    Ort::Value              input_tensor_{ nullptr };
    std::array<int64_t, 4>  input_shape_{ 1, 1, width_, height_ };
    Ort::Value              output_tensor_{ nullptr };
    std::array<int64_t, 2>  output_shape_{ 1, 10 };
};

void randomize_28x28_img(std::array<float, 784> &inp_array)
{
    float rand_num;
    for (int i = 0; i < inp_array.size(); ++i)
    {
        rand_num = distribution(generator);
        inp_array[i] = rand_num;
    }
}

int main(void)
{
    ONNX_Model model;
    std::vector<int64_t> dims = model.get_input_shape_from_session();
    std::cout << "Input Shape: (";
    std::cout << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << ")" << std::endl;
    
    std::ptrdiff_t result;
    for (int i = 0; i < 10; ++i)
    {
        randomize_28x28_img(model.input_image_);
        result = model.Run();
        std::cout << result << std::endl;
    }
    return 0;
}