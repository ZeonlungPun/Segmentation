/*
 * unetInference: A unet inference class for semantic segmentation
 * 
 * Author: Zeonlung Pun
 * Date: Jan 20, 2025
 * 
 * Description:
 * This class provides an implementation of unet algorithms inference 
 * using libtorch and OpenCV.
 * It includes methods for reading the model, preprocessing input images, running inference,
 * and drawing  results on the input image.
 * 
 * Usage:
 * The  class is instantiated with the .pt model path,
 * input image, and letter-box preprocessing and postprocessing.
 * The `main_process` function performs the complete inference and returns the processed image.
 *
 * Note:
 * This code assumes the libtorch and OpenCV libraries are properly installed and configured.
 * 
 * Contact: zeonlungpun@gmail.com
 * 
 */

#include<stdio.h>
#include<string>
#include<iostream>
#include<vector>
#include<tuple>
#include<opencv2/opencv.hpp>
#include<torch/torch.h>
#include<torch/script.h>

//predefine each class and its corrosponding colour

struct COLOR
{
	int B;
	int G;
	int R;
};

COLOR Classes[2]=
{
	{0,0,0},//background
	{128, 0, 0},

};


class unetInference
{
	/*
	Initializes an instance of the unetInference class.
	Args:
		model_path: Path to the .pt unet trained model.
		raw_img: the original input image.
		model_input_h/w: the input size of unet model
		result_save_path: the path to save the resulted image
	*/

public:
	std::string model_path;
	torch::jit::script::Module module;
	cv::Mat raw_img;
	int model_input_h;
	int model_input_w;
	int neww;
	int newh;
	int ori_w; 
	int ori_h;
	int num_classes;
	std::string result_save_path;

	unetInference(std::string model_path,cv::Mat raw_img,
	int model_input_h,int model_input_w,std::string result_save_path):
	model_path(model_path),raw_img(raw_img),model_input_h(model_input_h),
	model_input_w(model_input_w),result_save_path(result_save_path)
	{
		this->ori_h=raw_img.rows;
    	this->ori_w=raw_img.cols;
		this->neww=0;
		this->newh=0;
		this->num_classes=0;
		//load the .pt model
		try
		{
			this->module = torch::jit::load(model_path);
		}
		catch (const c10::Error& e)
		{
			std::cerr << "The model can't load\n";
			
		}
		printf("The model load success!\n");

	}

	cv::Mat PreprocessImage()
	{
		//letter-box image preprocess
		//original image size
		float iw=this->ori_w;
		float ih=this->ori_h;
		float scale= std::min( this->model_input_w/iw,this->model_input_h/ih);
		// eqaul ratio resize
		this->neww =static_cast<int>(iw*scale);
		this->newh =static_cast<int>(ih*scale);
		cv::Mat image;
		cv::resize(this->raw_img,image,cv::Size(this->neww,this->newh),cv::INTER_LINEAR);
		// target image size
		cv::Size model_inputsize(this->model_input_w,this->model_input_h);
		cv::Mat new_image(model_inputsize,CV_8UC3,cv::Scalar(128,128,128));
		// Paste the image into the new image at the calculated position
		image.copyTo(new_image(cv::Rect((this->model_input_w - this->neww) / 2, (this->model_input_h - this->newh) / 2, this->neww, this->newh)));


		return new_image;
	}

	torch::Tensor inference()
	{
		//preprocess the image
		// 1,letter box image
		cv::Mat newimage=PreprocessImage();
		//2, resize the image to (input_shape_h,input_shape_w,3)
		cv::Mat input;
		cv::resize(newimage, newimage, cv::Size(this->model_input_w, this->model_input_h));
		cv::cvtColor(newimage, input, cv::COLOR_BGR2RGB);

		// 3, convert cv::Mat to torch tensor
		torch::Tensor tensor_image = torch::from_blob(input.data, { 1,input.rows, input.cols,3 }, torch::kByte); 

		//4, shape->(batchsize,channles,w,h)
		tensor_image = tensor_image.permute({ 0,3,1,2 });
		tensor_image = tensor_image.toType(torch::kFloat);
		//5, normalize
		tensor_image = tensor_image.div(255);

		//set GPU and put the model into it
		torch::DeviceType *deviceType = new torch::DeviceType();
		if (torch::cuda::is_available())
		{
			*deviceType = torch::kCUDA;
			std::cout << "The cuda is available" << std::endl;
		}
		else
		{
			*deviceType = torch::kCPU;
			std::cout << "The cuda isn't available" << std::endl;
		}
		
		torch::Device device(*deviceType);
		std::cout << *deviceType << std::endl;


		try
		{
			this->module.to(device);
			std::cout << "put model into the " << device << std::endl;
		}
		catch (const c10::Error& e)
		{
			std::cerr << "The model into gpu faill\n";
			
		}

		//inference
		this->module.eval();

		torch::Tensor output;
		try
		{   //The shape is [batch_size, num_classes, input_shape_h,input_shape_w]
			output = module.forward({tensor_image.to(device)}).toTensor(); 
		}
		catch (const c10::Error& e)
		{
			std::cerr << "Can't get output!\n";
		
		}
		
		std::cout << "The output shape is: "<< output.sizes() << std::endl;

		return output;
	}

	torch::Tensor postprocess(torch::Tensor output)
	{
		//postprocess
		auto tmp = output[0]; 
		this->num_classes=output.sizes()[1];
		//[ num_classes, input_shape_h,input_shape_w]
		auto pred = torch::softmax(tmp.detach(), 0).cpu(); 
		
		int64_t start_h = (this->model_input_h - this->newh) / 2;
		int64_t start_w = (this->model_input_w - this->neww) / 2;
		torch::Tensor pr = pred.slice(1, start_h, start_h + this->newh)   
							.slice(2, start_w, start_w + this->neww);
		std::cout << "pr shape is " << pr.sizes() << std::endl;
		// [num_classes,input_shape_h,input_shape_w]
		std::cout << "pr shape is " << pr.sizes() << std::endl;
		//[num_classes,input_shape_h,input_shape_w]-->[num_classes,ori_h,ori_w]
		std::vector<int64_t> size = {this->ori_h, this->ori_w};
		torch::Tensor resized_pr = torch::nn::functional::interpolate(
			pr.unsqueeze(0),  // add batch axis
			torch::nn::functional::InterpolateFuncOptions()
				.size(size)  // target size
				.mode(torch::kBilinear)         
				.align_corners(false)            
		).squeeze(0);
		resized_pr =resized_pr.permute({1,2,0});
		//[ori_h,ori_w,num_classes] -->[ori_h,ori_w,1]
		resized_pr=resized_pr.argmax(-1);
		std::cout << "resized_pr shape is " << resized_pr.sizes() << std::endl;
		return resized_pr;
	}

	void Visualize(torch::Tensor resized_pr)
	{
		torch::Tensor seg_img = torch::zeros({ this->ori_h,this->ori_w,3 });
	
		for (int c = 0; c < num_classes; c++)
		{

			seg_img.index({ "...", 0 }) += ((resized_pr.index({ "..." }) == c) * Classes[c].B);
			seg_img.index({ "...", 1 }) += ((resized_pr.index({ "..."}) == c) * Classes[c].G);
			seg_img.index({ "...", 2 }) += ((resized_pr.index({ "..." }) == c) * Classes[c].R);

		}
		
		// convert the torch tensor to cv::Mat
		seg_img = seg_img.to(torch::kCPU).to(torch::kUInt8); 
		std::cout << "seq_img shape is " << seg_img.sizes() << std::endl;
		cv::Mat res(cv::Size(this->ori_w, this->ori_h), CV_8UC3,seg_img.data_ptr());
		
		cv::cvtColor(res,res,cv::COLOR_RGB2BGR);
		//std::cout << res << std::endl;
		cv::imwrite(this->result_save_path,res);

	}

	void main_process()
	{
		torch::Tensor output=this->inference();
		torch::Tensor resized_pr=postprocess(output);
		Visualize(resized_pr);


	}





};



int main()
{
    std::string model_path="/home/kingargroo/cpp/libtorchunet/model.pt";
    cv::Mat raw_img=cv::imread("/home/kingargroo/cpp/unet/test.jpg");
	std::string result_save_path="/home/kingargroo/cpp/libtorchunet/result.png";
    int model_input_h=640;
    int model_input_w=960;
	unetInference inference(model_path,raw_img,model_input_h,model_input_w,result_save_path);
	inference.main_process();
	
    return 0;
}