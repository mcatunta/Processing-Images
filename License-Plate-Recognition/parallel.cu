#include "ClsParallel.h"
#include <opencv2/core/core.hpp>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>

using namespace cv;
using namespace std;

namespace Parallel
{	
	#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)
	static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
	{
		if(err!=cudaSuccess)
		{
			fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
			std::cin.get();
			exit(EXIT_FAILURE);
		}
	}		
	
	__global__ void cu_gfilter_primero(unsigned char* imgray, unsigned char* temp, int width, int height, int grayWidthStep)
	{
		const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if((xIndex<width) && (yIndex<height))
		{
			const double coeffs[] = {0.0545, 0.2442, 0.4026, 0.2442, 0.0545};
			float sum = 0.0;
			int y1;
			for(int i = -2; i <= 2; i++){
				if (yIndex - i < 0)
					y1 = - (yIndex - i) - 1;
				else
					if (yIndex - i >= height)
						y1 = 2 * height - (yIndex - i) - 1;
					else
						y1 = yIndex - i;
				sum = sum + coeffs[i + 2] * imgray[y1 * grayWidthStep + xIndex];
			}
			temp[yIndex * grayWidthStep + xIndex] = static_cast<unsigned char>(sum);
		}
	}

	__global__ void cu_gfilter_segundo(unsigned char* temp, unsigned char* gfilter, int width, int height, int grayWidthStep)
	{
		const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if((xIndex<width) && (yIndex<height))
		{
			const double coeffs[] = {0.0545, 0.2442, 0.4026, 0.2442, 0.0545};
			float sum = 0.0;
			int x1;
			for(int i = -2; i <= 2; i++){
				if (xIndex - i < 0)
					x1 = - (xIndex - i) - 1;
				else
					if (xIndex - i >= width)
						x1 = 2 * width - (xIndex - i) - 1;
					else
						x1 = xIndex - i;
				sum = sum + coeffs[i + 2] * temp[yIndex * grayWidthStep + x1];
			}
			gfilter[yIndex * grayWidthStep + xIndex] = static_cast<unsigned char>(sum);
		}
	}

	__global__ void cu_canny_primero(unsigned char* imgray, unsigned char* m, unsigned char* ang, int width, int height, int grayWidthStep)
	{
		const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if((xIndex<width) && (yIndex<height))
		{
			const int gray_tid = (yIndex * grayWidthStep + xIndex);
			if ((xIndex == 0) || (xIndex == width-1) || (yIndex == 0) || (yIndex == height-1)) {
				m[gray_tid] = static_cast<unsigned char>(0);
				ang[gray_tid] = static_cast<unsigned char>(0);
			}
			else {
				const float dx = imgray[(yIndex - 1) * grayWidthStep + (xIndex + 1)] + 2 * imgray[(yIndex) * grayWidthStep + (xIndex + 1)] + imgray[(yIndex + 1) * grayWidthStep + (xIndex + 1)] - imgray[(yIndex - 1) * grayWidthStep + (xIndex - 1)] - 2 * imgray[(yIndex) * grayWidthStep + (xIndex - 1)] - imgray[(yIndex + 1) * grayWidthStep + (xIndex - 1)];
				const float dy = imgray[(yIndex - 1) * grayWidthStep + (xIndex - 1)] + 2 * imgray[(yIndex - 1) * grayWidthStep + (xIndex)] + imgray[(yIndex - 1) * grayWidthStep + (xIndex + 1)] - imgray[(yIndex + 1) * grayWidthStep + (xIndex - 1)] - 2 * imgray[(yIndex + 1) * grayWidthStep + (xIndex)] - imgray[(yIndex + 1) * grayWidthStep + (xIndex + 1)];
				const int magnitud= static_cast<int>(abs(dy) + abs(dx));
				const int angulo = static_cast<int>((atan(dy/dx))*57.33);
				m[gray_tid] = static_cast<unsigned char>(magnitud);//(sqrt((dx*dx) + (dy*dy)));
				ang[gray_tid] = static_cast<unsigned char>(angulo);//(360)/(2*3.14));
				//m[gray_tid] = static_cast<unsigned char>(static_cast<int>(abs(dx) + abs(dy)));//(sqrt((dx*dx) + (dy*dy)));
				//ang[gray_tid] = static_cast<unsigned char>(atan(dx/dy)*57.33);
			}
		}
	}

	__global__ void cu_canny_segundo(unsigned char* m, unsigned char* ang, unsigned char* borde, unsigned char* out, int width, int height, int grayWidthStep)
	{
		const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if((xIndex<width) && (yIndex<height))
		{
			const int tid  = (yIndex * grayWidthStep + xIndex);

			if ((xIndex == 0) || (xIndex == width-1) || (yIndex == 0) || (yIndex == height-1)) {				
				borde[tid] = static_cast<unsigned char>(0);
			}
			else {
				if(ang[tid] = 0)//>= 0 && ang[tid] < 45)
				{
					if ((m[tid] > m[(yIndex) * grayWidthStep + (xIndex + 1)]) && (m[tid] > m[(yIndex) * grayWidthStep + (xIndex - 1)]))
						borde[tid] = static_cast<unsigned char>(m[tid]);
					else
						borde[tid] = static_cast<unsigned char>(0);
				}
				else if(ang[tid] = 45)//>= 45 && ang[tid] < 90)
				{
					if ((m[tid] > m[(yIndex + 1) * grayWidthStep + (xIndex - 1)]) && (m[tid] > m[(yIndex - 1) * grayWidthStep + (xIndex + 1)]))
						borde[tid] = static_cast<unsigned char>(m[tid]);
					else
						borde[tid] = static_cast<unsigned char>(0);
				}
				else if(ang[tid] = 90)//>= 90 && ang[tid] < 135)
				{
					if ((m[tid] > m[(yIndex + 1) * grayWidthStep + (xIndex)]) && (m[tid] > m[(yIndex - 1) * grayWidthStep + (xIndex)]))
						borde[tid] = static_cast<unsigned char>(m[tid]);
					else
						borde[tid] = static_cast<unsigned char>(0);						
				}
				else if(ang[tid] = 135)//>= 135 && ang[tid] < 180)
				{
					if ((m[tid] > m[(yIndex + 1) * grayWidthStep + (xIndex + 1)]) && (m[tid] > m[(yIndex - 1) * grayWidthStep + (xIndex - 1)]))
						borde[tid] = static_cast<unsigned char>(m[tid]);
					else
						borde[tid] = static_cast<unsigned char>(0);						
				}
				else
					borde[tid] = static_cast<unsigned char>(0);
				if (borde[tid] > 120)
					out[tid] = static_cast<unsigned char>(1);
				else {
					if (borde[tid] <= 40)
						out[tid] = static_cast<unsigned char>(0);
					else
						out[tid] = static_cast<unsigned char>(2);
				}
			}
		}
	}

	__global__ void cu_canny_tercero(unsigned char* out, unsigned char* out2, int width, int height, int grayWidthStep)
	{
		const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if((xIndex<width) && (yIndex<height))
		{
			const int tid  = yIndex * grayWidthStep + xIndex;

			if (out[tid] == 2) {
				if ((out[(yIndex - 1) * grayWidthStep + (xIndex - 1)] == 1) || (out[(yIndex - 1) * grayWidthStep + (xIndex)] == 1)
					|| (out[(yIndex - 1) * grayWidthStep + (xIndex + 1)] == 1) || (out[(yIndex) * grayWidthStep + (xIndex - 1)] == 1)
					|| (out[(yIndex) * grayWidthStep + (xIndex + 1)] == 1) || (out[(yIndex + 1) * grayWidthStep + (xIndex - 1)] == 1)
					|| (out[(yIndex + 1) * grayWidthStep + (xIndex)] == 1) || (out[(yIndex + 1) * grayWidthStep + (xIndex + 1)] == 1)) {
						out2[tid] = static_cast<unsigned char>(1);
				}
				else {
					out2[tid] = static_cast<unsigned char>(0);
				}
			}
			else{
				out2[tid] = static_cast<unsigned char>(out[tid]);
			}
		}
	}
		
	__global__ void cu_svm_primero(unsigned char* borde, float* output,int width, int height, int WidthStep, const float* vs, int widthsmall, int heightsmall, double rho)
	{
		const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if((xIndex<width) && (yIndex<height))
		{
			const int tid  = yIndex * WidthStep + (xIndex);
			if ((xIndex<width - widthsmall + 1) && (yIndex<height - heightsmall + 1))
			{
				float s = 0;
				for (int f = 0; f < heightsmall; f++) {
					for (int g = 0; g < widthsmall; g++) {
						s += borde[(f + yIndex) * WidthStep + (g + xIndex)] * vs[(f*widthsmall + g)];
					}
				}
				s = -1*(s + rho);				
				if (s>0)
					output[tid] = s;
				else
					output[tid] = -2;
			}
			else
				output[tid] = -1;
		}
	}
	
	__global__ void cu_convert_gray(unsigned char* imcolor, unsigned char* imgray, int width, int height, int colorWidthStep, int grayWidthStep)
	{
		const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if((xIndex<width) && (yIndex<height))
		{
			const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
			const int gray_tid  = yIndex * grayWidthStep + xIndex;
			const unsigned char blue	= imcolor[color_tid];
			const unsigned char green	= imcolor[color_tid + 1];
			const unsigned char red		= imcolor[color_tid + 2];
			const float gray = red * 0.2989f + green * 0.5870f + blue * 0.1140f;
			imgray[gray_tid] = static_cast<unsigned char>(gray);
		}
	}
	
	int getIndex(int M, int x){
		if(x < 0)
			return -x - 1;
		if(x >= M)
			return 2*M - x - 1;
		return x;
	}

	Mat gaussianFilter(Mat image){
		Mat out, temp;
		int x,y,i;
		float sum, x1,y1;

		if( !image.data )
			return image;

		double coeffs[] = {0.0545, 0.2442, 0.4026, 0.2442, 0.0545};

		out = image.clone();
		temp = image.clone();
		for(y = 0; y < image.rows; y++){
			  for(x = 0; x < image.cols; x++){
				  sum = 0.0;
				  for(i = -2; i <= 2; i++){
					  y1 = getIndex(image.rows, y - i);
					  sum = sum + coeffs[i + 2]*image.at<uchar>(y1, x);
				  }
				  temp.at<uchar>(y,x) = sum;
			  }
		}
		for(y = 0; y < image.rows; y++){
			  for(x = 0; x < image.cols; x++){
				  sum = 0.0;
				  for(i = -2; i <= 2; i++){
					  x1 = getIndex(image.cols, x - i);
					  sum = sum + coeffs[i + 2]*temp.at<uchar>(y, x1);
				  }
				  out.at<uchar>(y,x) = sum;
			  }
		}
		return out;
	}

	Mat canny(Mat im){
		Mat imgray(im.size(), CV_8U);
		Mat out(imgray.size(), CV_8U);
		const dim3 block(16,16);
		const dim3 grid((im.cols + block.x - 1)/block.x, (im.rows + block.y - 1)/block.y);
		const int colorBytes = im.step * im.rows;
		const int grayBytes = imgray.step * imgray.rows;
		unsigned char *d_imcolor, *d_imgray;
		SAFE_CALL(cudaMalloc<unsigned char>(&d_imcolor,colorBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_imgray,grayBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMemcpy(d_imcolor,im.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		cu_convert_gray<<<grid,block>>>(d_imcolor,d_imgray,im.cols,im.rows,im.step,imgray.step);
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
		SAFE_CALL(cudaMemcpy(imgray.ptr(),d_imgray,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");	
		SAFE_CALL(cudaFree(d_imcolor),"CUDA Free Failed");
		unsigned char *d_temp, *d_gfilter, *d_m, *d_ang;		
		SAFE_CALL(cudaMalloc<unsigned char>(&d_temp,grayBytes),"CUDA Malloc Failed");
		cu_gfilter_primero<<<grid,block>>>(d_imgray,d_temp,imgray.cols,imgray.rows,imgray.step);
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
		SAFE_CALL(cudaFree(d_imgray),"CUDA Free Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_gfilter,grayBytes),"CUDA Malloc Failed");
		cu_gfilter_segundo<<<grid,block>>>(d_temp,d_gfilter,imgray.cols,imgray.rows,imgray.step);
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
		SAFE_CALL(cudaFree(d_temp),"CUDA Free Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_m,grayBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_ang,grayBytes),"CUDA Malloc Failed");
		cu_canny_primero<<<grid,block>>>(d_gfilter,d_m,d_ang,imgray.cols,imgray.rows,imgray.step);		
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
		SAFE_CALL(cudaFree(d_gfilter),"CUDA Free Failed");		
		unsigned char *d_borde, *d_out, *d_out2;
		SAFE_CALL(cudaMalloc<unsigned char>(&d_borde,grayBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_out,grayBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_out2,grayBytes),"CUDA Malloc Failed");
		cu_canny_segundo<<<grid,block>>>(d_m,d_ang,d_borde,d_out,imgray.cols,imgray.rows,imgray.step);
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");	
		SAFE_CALL(cudaFree(d_m),"CUDA Free Failed");
		SAFE_CALL(cudaFree(d_ang),"CUDA Free Failed");
		SAFE_CALL(cudaFree(d_borde),"CUDA Free Failed");
		cu_canny_tercero<<<grid,block>>>(d_out,d_out2,imgray.cols,imgray.rows,imgray.step);
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
		SAFE_CALL(cudaMemcpy(out.ptr(),d_out2,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");		
		SAFE_CALL(cudaFree(d_out),"CUDA Free Failed");
		SAFE_CALL(cudaFree(d_out2),"CUDA Free Failed");
		Mat borde = Mat::zeros(imgray.size(), imgray.type());
		imgray.copyTo(borde, out);
		return borde;	
	}

	///// SVM UBICAR PLACA

	const int ndatab=33;//35;//8;
	const int ndatam=33;//35;//8;
	const int alto=50;//50;//75;//100;//25;//100;//50;
	const int ancho=90;//90;//195;//49;//195;//190;

	Mat trainingdata(ndatab+ndatam, alto*ancho, CV_32FC1), traininglabel(ndatab+ndatam, 1, CV_32FC1);

	Mat ocupar(Mat o, int x, int y){
		Mat p=o.clone();
		for (int i = x; i < alto+x; i++) {
			for (int j = y; j < ancho+y; j++) {
				if ((i == x) || (j == y) || (i == alto+x-1) || (j == ancho+y-1))
					p.at<uchar>(i,j) = 128;
				else
					p.at<uchar>(i,j) = 255;
			}
		}
		return p;
	}

	void loadData(){
		Mat imb, imm;
		string inb, inm;
		stringstream num;
		int c, k;
		for (k = 1; k <= ndatab; k++) {
			num<<k;
			inb = "D://images/positives/" + num.str() + ".jpg";
			imb = imread(inb);
			c = 0;
			for (int i = 0; i < imb.rows; i++) {
				for (int j = 0; j < imb.cols; j++) {
					trainingdata.at<float>(k-1,c) = (float)imb.at<uchar>(i,j);
					traininglabel.at<float>(k-1,0) = 1.0;
					c++;
				}
			}
			num.str("");
			num.clear();
		}
		for (k = 1; k <= ndatam; k++) {
			num<<k;
			inm = "D://images/negatives/" + num.str() + ".jpg";
			imm = imread(inm);
			c = 0;
			for (int i = 0; i < imm.rows; i++) {
				for (int j = 0; j < imm.cols; j++) {
					trainingdata.at<float>(k-1+ndatab,c) = (float)imm.at<uchar>(i,j);
					traininglabel.at<float>(k-1+ndatab,0) = -1.0;					
					c++;
				}
			}
			num.str("");
			num.clear();
		}
	}

	Mat paint(Mat o, int x, int y){
		Mat p=o.clone();
		for (int i = x; i < alto+x; i++) {
			for (int j = y; j < ancho+y; j++) {
				if ((i == x) || (j == y) || (i == alto+x-1) || (j == ancho+y-1))
					p.at<Vec3b>(i, j) = 128;
			}
		}
		return p;
	}

	class mySVM : public CvSVM
	{
	public:
		double getRho();
	};

	double mySVM::getRho()
	{
		const CvSVMDecisionFunc *dec = CvSVM::decision_func;		
		return (- dec[0].rho);
	}

	void find(Mat original_color, string path){
		Mat im = original_color;
		Mat borde = canny(im);

		CvSVMParams param = CvSVMParams();
		param.svm_type    = CvSVM::C_SVC;
		param.kernel_type = CvSVM::LINEAR;
		param.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		loadData();

		mySVM svm;
		svm.train(trainingdata, traininglabel, cv::Mat(), cv::Mat(), param);
		const float* vs = svm.get_support_vector(0);
		double rho = svm.getRho();		
		const dim3 block(16,16);
		const dim3 grid((borde.cols + block.x - 1)/block.x, (borde.rows + block.y - 1)/block.y);
		const int bordeBytes = borde.step * borde.rows;
		const int outputBytes = borde.cols * borde.rows * sizeof(float);
		const int vsBytes = ancho * alto * sizeof(float);
		float* h_output = new float[borde.cols * borde.rows];
		unsigned char *d_borde;
		float * d_output, *d_vs;
		SAFE_CALL(cudaMalloc(&d_borde,bordeBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc(&d_output,outputBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc(&d_vs,vsBytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMemcpy(d_borde,borde.ptr(),bordeBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaMemcpy(d_vs,vs,vsBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		cu_svm_primero<<<grid,block>>>(d_borde,d_output,borde.cols,borde.rows,borde.step,d_vs,ancho,alto,rho);
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
		SAFE_CALL(cudaMemcpy(h_output,d_output,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaFree(d_borde),"CUDA Free Failed");
		SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
		SAFE_CALL(cudaFree(d_vs),"CUDA Free Failed");
		
		float tope=0; int ax = -1,ay = -1;
		for (int i = 0; i < borde.rows; i++) {
			for (int j = 0; j < borde.cols; j++) {
				float tem = h_output[j + i*(borde.cols)];
				if ( tem > tope) {
					tope = tem;
					//ocupado = ocupar(ocupado, i, j);
					ax=i;ay=j;
				}
			}
		}
		//ocupadolabel = ocupar(ocupadolabel, ax, ay);
		
		if (ax != -1){
			original_color = paint(original_color, ax, ay);
		}
		imshow("Output", original_color);
		return;
	}	

	void findInVideo(string path){
		VideoCapture cap(path.substr(0,path.length()-6) + "video.mp4");
		if(!cap.isOpened())
			return;
		Mat edges;
		stringstream ni;
		int i=1,j=1;
		for (;;)
		{
			Mat frameVideo;
			cap>>frame;
			if (j==3)
			{				
				if ( frameVideo.empty() ) 
				return;	
				Mat frameImage = Mat(frameVideo, cv::Rect(500,450,800,400));
				find(frameImage, path);
				if (waitKey(30) == 13)
				{
					ni << i;
					string tempath = path.substr(0,path.length()-4) + "c" + ni.str() + ".jpg";
					imwrite(tempath, frameImage);
					i=i+1;
					ni.str("");
					ni.clear();
				}
				j=1;				
			}
			else{
				j=j+1;
			}
		}		
		return;
	}

	int findInImage(string path){
		try {
			Mat im = imread(path, CV_LOAD_IMAGE_COLOR);
			if ( im.empty() ) 
				return -1;
			find(im, path);
			return 1;
		} catch (const std::bad_alloc& e) {
			std::cout << "Allocation failed: " << e.what() << '\n';
		}		
	}

	void convertMasive(){
		string in, out;
		stringstream num;
		for(int i=1;i<ndatab+1;++i){
			Mat im, borde;
			num<<i;
			in = "D://images/color/positives" + num.str() + ".jpg";
			im = imread(in, CV_LOAD_IMAGE_COLOR);
			borde = canny(im);
			out = "D://images/gray/positives" + num.str() + ".jpg";
			imwrite(out, borde);
			num.str("");
			num.clear();
		}
		for(int i=1;i<ndatam+1;++i){
			Mat im, borde;
			num<<i;
			in = "D://images/color/negatives" + num.str() + ".jpg";
			im = imread(in, CV_LOAD_IMAGE_COLOR);
			borde = canny(im);
			out = "D://images/gray/negatives" + num.str() + ".jpg";
			imwrite(out, borde);
			num.str("");
			num.clear();
		}		
	}
}