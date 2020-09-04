#include "ClsSecuencial.h"
#include <opencv2/core/core.hpp>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"

using namespace cv;
using namespace std;

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
		Mat ca=im.clone();
		ca = gaussianFilter(ca);
		Mat m = Mat::zeros(ca.size(), CV_8U);
		Mat ang = Mat::zeros(ca.size(), CV_8U);
		Mat borde = Mat::ones(ca.size(), CV_8U);
		Mat out = -1*Mat::ones(ca.size(), CV_8U);
		float dx=0, dy=0;
		for (int x = 0; x < im.rows; x++) {
			for (int y = 0; y < im.cols; y++) {
				if ((x == 0) || (x == im.rows-1) || (y == 0) || (y == im.cols-1)) {
					m.at<uchar>(x,y) = 0;
					ang.at<uchar>(x,y) = 0;
				}
				else {
					dx = (ca.at<uchar>(x-1,y+1) + 2*ca.at<uchar>(x,y+1) + ca.at<uchar>(x+1,y+1) - ca.at<uchar>(x-1,y-1) - 2*ca.at<uchar>(x,y-1) - ca.at<uchar>(x+1,y-1));
					dy = (ca.at<uchar>(x-1,y-1) + 2*ca.at<uchar>(x-1,y) + ca.at<uchar>(x-1,y+1) - ca.at<uchar>(x+1,y-1) - 2*ca.at<uchar>(x+1,y) - ca.at<uchar>(x+1,y+1));
					m.at<uchar>(x,y) = (int)((abs(dx) + abs(dy)));//sqrt(dx*dx + dy*dy);
					ang.at<uchar>(x,y) = (int)((atan(dy/dx)*(57.33)));//(360)/(2*3.14));
				}
			}
		}
		for (int x = 0; x < im.rows; x++) {
			for (int y = 0; y < im.cols; y++) {
				if ((x == 0) || (x == im.rows-1) || (y == 0) || (y == im.cols-1))
					borde.at<uchar>(x,y) = 0;
				else {
					if(ang.at<uchar>(x,y) = 0)// && ang.at<uchar>(x,y) < 45)
					{
						if ((m.at<uchar>(x,y) > m.at<uchar>(x,y+1)) && (m.at<uchar>(x,y) > m.at<uchar>(x,y-1)))
							borde.at<uchar>(x,y) = m.at<uchar>(x,y);
						else
							borde.at<uchar>(x,y) = 0;
					}
					else if(ang.at<uchar>(x,y) =45)//>= 45 && ang.at<uchar>(x,y) < 90)
					{
						if ((m.at<uchar>(x,y) > m.at<uchar>(x+1,y-1)) && (m.at<uchar>(x,y) > m.at<uchar>(x-1,y+1)))
							borde.at<uchar>(x,y) = m.at<uchar>(x,y);
						else
							borde.at<uchar>(x,y) = 0;
					}
					else if(ang.at<uchar>(x,y) =90)//>= 90 && ang.at<uchar>(x,y) < 135)
					{
						if ((m.at<uchar>(x,y) > m.at<uchar>(x+1,y)) && (m.at<uchar>(x,y) > m.at<uchar>(x-1,y)))
							borde.at<uchar>(x,y) = m.at<uchar>(x,y);
						else
							borde.at<uchar>(x,y) = 0;						
					}
					else if(ang.at<uchar>(x,y) =135)//>= 135 && ang.at<uchar>(x,y) < 180)
					{
						if ((m.at<uchar>(x,y) > m.at<uchar>(x+1,y+1)) && (m.at<uchar>(x,y) > m.at<uchar>(x-1,y-1)))
							borde.at<uchar>(x,y) = m.at<uchar>(x,y);
						else
							borde.at<uchar>(x,y) = 0;
					}
					else
						borde.at<uchar>(x,y) = 0;
				}
				if (borde.at<uchar>(x,y) > 120)
					out.at<uchar>(x,y) = 1;
				else {
					if (borde.at<uchar>(x,y) <= 40)
						out.at<uchar>(x,y) = 0;
					else
						out.at<uchar>(x,y) = 2;
				}
			}
		}
		for (int x = 0; x < im.rows; x++) {
			for (int y = 0; y < im.cols; y++) {
				if (out.at<uchar>(x,y) == 2) {
					if ((out.at<uchar>(x-1,y-1) == 1) || (out.at<uchar>(x-1,y) == 1)
							|| (out.at<uchar>(x-1,y+1) == 1) || (out.at<uchar>(x,y-1) == 1)
							|| (out.at<uchar>(x,y+1) == 1) || (out.at<uchar>(x+1,y-1) == 1)
							|| (out.at<uchar>(x+1,y) == 1) || (out.at<uchar>(x+1,y+1) == 1)) {
						out.at<uchar>(x,y) = 1;
					}
					else {
						out.at<uchar>(x,y) = 0;
					}
				}
			}
		}
		return out;
	}

	///// SVM UBICAR PLACA

	const int ndatab=33; // Cantidad de imagenes de placas
	const int ndatam=33; // Cantidad de imagenes sin placas
	const int alto=50;   // Alto de la ventana de busqueda
	const int ancho=90;  // Ancho de la ventana de busqueda

	Mat trainingData(ndatab+ndatam, alto*ancho, CV_32FC1), traininglabel(ndatab+ndatam, 1, CV_32FC1);

	Mat fill(Mat o, int x, int y){
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
			inb = "D://images/positives/" + num.str() + ".jpg"; // Ubicacion de las imagenes de prueba positivas
			imb = imread(inb);
			c = 0;
			for (int i = 0; i < imb.rows; i++) {
				for (int j = 0; j < imb.cols; j++) {
					trainingData.at<float>(k-1,c) = (float)imb.at<uchar>(i,j);
					traininglabel.at<float>(k-1,0) = 1.0;
					c++;
				}
			}
			num.str("");
			num.clear();
		}
		for (k = 1; k <= ndatam; k++) {
			num<<k;
			inm = "D://images/negatives/" + num.str() + ".jpg"; // Ubicacion de las imagenes de prueba negativas
			imm = imread(inm);
			c = 0;
			for (int i = 0; i < imm.rows; i++) {
				for (int j = 0; j < imm.cols; j++) {
					trainingData.at<float>(k-1+ndatab,c) = (float)imm.at<uchar>(i,j);
					traininglabel.at<float>(k-1+ndatab,0) = -1.0;					
					c++;
				}
			}
			num.str("");
			num.clear();
		}
	}

	Mat convertToTestData(Mat o, int a, int b){
		Mat t = Mat(1, alto*ancho, CV_32FC1);
		int c=0;
		for (int i = a; i < alto+a; i++) {
			for (int j = b; j < ancho+b; j++) {
				t.at<float>(0,c) = (float)o.at<uchar>(i,j);
				c++;
			}
		}		
		return t;
	}

	Mat paint(Mat o, int x, int y){
		Mat p=o.clone();
		for (int i = x; i < alto+x; i++) {
			for (int j = y; j < ancho+y; j++) {
				if ((i == x) || (j == y) || (i == alto+x-1) || (j == ancho+y-1))
					p.at<Vec3b>(i,j) = 128;
			}
		}
		return p;
	}

	void find(Mat imgray, Mat im, string path){
		float r;
		Mat subim;
		int ctemp = 0, nlabel=0;
		Mat imout=im.clone();
		Mat test;
		Mat ocupado = Mat::zeros(imgray.size(), CV_8U);
		Mat ocupadolabel = Mat::zeros(imgray.size(), CV_8U);

		CvSVMParams param = CvSVMParams();
		param.svm_type    = CvSVM::C_SVC;
		param.kernel_type = CvSVM::LINEAR;
		param.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		cargarData();

		CvSVM svm(trainingData, traininglabel, cv::Mat(), cv::Mat(), param);
	
		float tope=0; int ax,ay;
		for (int i = 0; i < imgray.rows - alto + 1; i++) {
			for (int j = 0; j < imgray.cols - ancho + 1; j++) {
				test = Mat(1, alto * ancho, CV_32FC1);
				test = convertToTestData(imgray,i, j);
				r = svm.predict(test);
				if (r==1){
					ocupado = fill(ocupado, i, j);
					r=-1*svm.predict(test,true);
					if (r>tope){
						tope = r;
						ax = i; ay = j;
					}
				}
			}
		}
		ocupadolabel = fill(ocupadolabel, ax, ay);
		imout = paint(imout, ax, ay);
		string tempath = path.substr(0,path.length()-4) + "secuencial.jpg";
		imwrite(tempath, imout);
		imshow("Output with filter",ocupado);
		imshow("Output without filter",ocupadolabel);
		return;
	}

	Mat convertToGray(Mat im){
		Mat imgray;
		imgray.create(im.size(), CV_8U);
		for (int i = 0; i < im.cols; i++) {
			for (int j = 0; j < im.rows; j++) {
				Vec3b intensity = im.at<Vec3b>(j, i);
				imgray.at<uchar>(j,i) = (intensity.val[0]*0.1140 + intensity.val[1] *0.5870+ intensity.val[2]*0.2989);
			}
		}
		return imgray;
	}

	int CPL::Secuencial::Buscar(string path){
		try {
			Mat original_color = imread(path, CV_LOAD_IMAGE_COLOR);
		if ( original_color.empty() ) 
        	return -1; 
		Mat imgray;
		Mat borde, out;
		Mat im = original_color;
		imgray = convertToGray(im);
		out.create(imgray.size(), imgray.type());
		borde = canny(imgray);
		out = Scalar::all(0);
		imgray.copyTo(out, borde);
		find(out, im, path);
		imshow("Output",out);
		return 1;
		} catch (const std::bad_alloc& e) {
			std::cout << "Allocation failed: " << e.what() << '\n';
		}		
	}

	void CPL::Secuencial::convertMasive(){
		try {
			Mat im, imgray, outf, borde;
			string in, out;
			stringstream num;
			for(int i=1;i<ndatab+1;++i){
				num<<i;
				in = "D://images/color/positives" + num.str() + ".jpg"; 
				im = imread(in, CV_LOAD_IMAGE_COLOR);
				imgray = convertToGray(im);
				borde = canny(imgray);				
				outf.create(imgray.size(), imgray.type());
				outf = Scalar::all(0);
				imgray.copyTo(outf, borde);
				out = "D://images/gray/positives" + num.str() + ".jpg";
				imwrite(out, outf);
				num.str("");
				num.clear();
			}
			for(int i=1;i<ndatam+1;++i){
				num<<i;
				in = "D://images/color/negatives" + num.str() + ".jpg";
				im = imread(in, CV_LOAD_IMAGE_COLOR);
				imgray = convertToGray(im);
				borde = canny(imgray);				
				outf.create(imgray.size(), imgray.type());
				outf = Scalar::all(0);
				imgray.copyTo(outf, borde);
				out = "D://images/gray/negatives" + num.str() + ".jpg";
				imwrite(out, outf);
				num.str("");
				num.clear();
			}
		} catch (const std::bad_alloc& e) {
			std::cout << "Allocation failed: " << e.what() << '\n';
		}
		
	}
