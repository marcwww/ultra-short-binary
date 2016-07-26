#include "usb_ft.h"


int main()
{
	cv::Mat img_1=cv::imread("1.jpg",0);
	cv::Mat img_2=cv::imread("2.jpg",0);

	usb_ft::usb ft_1(img_1);
	usb_ft::usb ft_2(img_2);

	//double t=cv::getTickCount();
	ft_1.generate_ft();
	//t=(cv::getTickCount()-t)/cv::getTickFrequency();
	//std::cout<<t<<"s"<<std::endl;

	ft_2.generate_ft();
	
	std::vector<int> res;

	int num=usb_ft::match(ft_1,ft_2,res);

	std::vector<cv::DMatch> match_res(num);
	int j=0;
	for(int i=0;i<res.size();i++)
		if(res[i]!=-1)// -1 means there is no match between this pair.
			match_res[j++]=cv::DMatch(i,res[i],0);
	
	cv::Mat res_img;
	cv::drawMatches(img_1,ft_1.kps,img_2,ft_2.kps,match_res,res_img,cv::Scalar(255,0,0));

	cv::imshow("res",res_img);
	cv::waitKey(0);

	//ft_1.show_kps("1");
	//ft_2.show_kps("2");

	
	return 0;
}

