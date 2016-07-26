#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp> 
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <omp.h>


namespace usb_ft
{
	namespace usb_params
	{
		extern int nOctives;
		extern int D;
		extern int g;
		extern int alpha;
		extern int beta;
		extern int C;
		extern int n;
		extern float omega;
		extern int d;
		extern int m;
		extern int phi;
		extern int match_order;
		extern int gama;
		extern int eta;
		extern int mu;
	}

	class usb
	{
	public:
		usb(const cv::Mat& src_img):src(src_img)
		{
		}

		void generate_ft();
		void show_kps(const char* win_name);

		class asf
		{
		public:

			struct val
			{
				cv::Mat v;
				int ori;
				int dis;
				
				val()
				{
					v=cv::Mat(8,1,CV_8U);	
				}
				val(const cv::Mat& v,int ori,int dis)
				{
					this->v=v;
					this->ori=ori;
					this->dis=dis;
				}
				val(const val& va)
				{
					this->v=va.v;
					this->ori=va.ori;
					this->dis=va.dis;
				}
			};
			std::vector<val> vals;
			std::vector<int> indices;


			asf()
			{
			}

			asf(const asf& a)
			{
				this->vals=a.vals;
				this->indices=a.indices;
			}
			
			void push_back(int index,const cv::Mat& v,int ori,int dis)
			{
				indices.push_back(index);
				vals.push_back(val(v,ori,dis));
			}

		};


		std::vector<cv::KeyPoint> kps;
		cv::Mat M;

		std::vector<asf> asfs;// sharing the index with kps.
		std::vector<cv::Mat> usbs;// sharing the index with kps.

		static int kp_match(const cv::Mat& Ua,const cv::Mat& Ub,const asf& Sa,const asf& Sb);
		//Return match order, from 0 to 4.

	private:
		cv::Mat src;
		std::vector<cv::Point2f> kps_src;
		cv::flann::Index* kps_kdtree;

	
		
		
		
		/*static int nOctives;
		static int D;
		static int g;
		static int alpha;
		static int beta;
		static int C;
		static int n;
		static float omega;
		static int d;*/
		

		static cv::Mat haar[5];
		static bool is_haar_loaded;

		void calc_M_col(const cv::Mat& sub_img,int i_start,int j);
		void M_bin_selection();
		void split_M();

		float MD(int i);
		float CR(int i,int j);


		bool calc_haar(const cv::Mat& sub_img,int i);

		void usb_extraction();
		void asf_extraction();

		class order_kps
		{
		public:
			bool operator()(const cv::KeyPoint& a,const cv::KeyPoint& b) const
			{
				return a.response>b.response;
			}
		};

		
	};

	int match(const usb& a, const usb& b, std::vector<int>& res);// return num of matches
}