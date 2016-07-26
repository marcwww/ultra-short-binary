#include "usb_ft.h"

namespace usb_ft
{
	cv::Mat usb::haar[5];
	/*int usb::D=36;
	int usb::g=3;
	int usb::alpha=12;
	int usb::beta=12;
	int usb::C=4;
	int usb::nOctives=4;
	int usb::n=10;
	float usb::omega=0.35;
	int usb::d=24;*/


	int usb_params::nOctives=4;
	int usb_params::D=36;
	int usb_params::g=3;
	int usb_params::alpha=12;
	int usb_params::beta=12;
	int usb_params::C=4;
	int usb_params::n=100;
	float usb_params::omega=0.35;
	int usb_params::d=24;
	int usb_params::m=16;
	int usb_params::phi=5;
	int usb_params::match_order=1;
	int usb_params::gama=2;
	int usb_params::eta=2;
	int usb_params::mu=3;




	bool usb::is_haar_loaded=false;

	void usb::generate_ft()
	{
		
		cv::SURF surf_ft;
		surf_ft(src,cv::noArray(),kps,cv::noArray());
		

		//cv::SIFT s_ft;
		
		//s_ft(src,cv::noArray(),kps,cv::noArray());
		//double t=cv::getTickCount();
		std::sort(kps.begin(),kps.end(),order_kps());
//		t=(cv::getTickCount()-t)/cv::getTickFrequency();
		//std::cout<<t<<"s"<<std::endl;


		std::vector<cv::KeyPoint> temp_container;
		int collection_size=0;

		omp_set_num_threads(8);
#pragma opm parallel for
		for(int i=0;i<kps.size();i++)
		{
			//Omit the key points lying close to the edges of the image.
			try
			{
				int wp=kps[i].size*usb_params::alpha;
				cv::RotatedRect sub_rect(kps[i].pt,cv::Size2f(wp,wp),kps[i].angle);
				cv::Rect a=sub_rect.boundingRect();
				if(!(a.x>=0 && a.y>=0 && a.x+a.width<src.cols && a.y+a.width<src.rows))
					continue;

				cv::Mat sub_bimg(src,sub_rect.boundingRect());
			}
			catch(cv::Exception e)
			{
				//std::cout<<e.err<<std::endl;
				continue;
			}

			int is_existed=false;
			for(int j=0;j<temp_container.size();j++)
				if((kps[i].pt.x==temp_container[j].pt.x) && (kps[i].pt.y==temp_container[j].pt.y))
				{
					is_existed=true;
					break;
				}
			if(!is_existed)
			{
				temp_container.push_back(kps[i]);
				collection_size++;
				if(collection_size==usb_params::n)
					break;
			}
		}

		kps=temp_container;

		// Initial kd tree.
		for(int i=0;i<kps.size();i++)
			kps_src.push_back(kps[i].pt);
		cv::flann::KDTreeIndexParams index_params(4);
		kps_kdtree=new cv::flann::Index(cv::Mat(kps_src).reshape(1),index_params);
		



		M=cv::Mat(45,kps.size(),CV_8U);



		usb_extraction();
		asf_extraction();
	
	}

	void usb::usb_extraction()
	{
		std::vector<cv::Mat> sub_imgs;
		
		for(int i=0;i<kps.size();i++)
		{
			int wp=kps[i].size*usb_params::alpha;
			
			try
			{
				cv::RotatedRect sub_rect(kps[i].pt,cv::Size2f(wp,wp),kps[i].angle);
				
				//cv::Point2f vertices[4];
				//sub_rect.points(vertices);
				//for (int i = 0; i < 4; i++)
					//line(src, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));

				cv::Rect a=sub_rect.boundingRect();

				cv::Mat sub_bimg(src,sub_rect.boundingRect());
				//cv::imshow("origin",sub_bimg);

				cv::Mat rotated_mat=cv::getRotationMatrix2D(cv::Point2f(sub_bimg.cols/2,sub_bimg.rows/2),kps[i].angle,1);

				cv::Mat upright_sub_bimg;
				cv::warpAffine(sub_bimg,upright_sub_bimg,rotated_mat,sub_bimg.size());
				
				cv::Mat upright_sub_img(upright_sub_bimg,cv::Rect(upright_sub_bimg.cols/2-wp/2,upright_sub_bimg.rows/2-wp/2,wp,wp));

				cv::Mat normalized_path;
				cv::resize(upright_sub_img,normalized_path,cv::Size(usb_params::D,usb_params::D));

				//cv::imshow("normalized",normalized_path);
				//cv::imshow("src",src);
				//cv::waitKey(1);

				
				//si_index: sub_image index
				for(int si_index=0;si_index<usb_params::g*usb_params::g;si_index++)
				{
					int step=usb_params::D/usb_params::g;

					cv::Rect x((si_index*step)%usb_params::D,(si_index/usb_params::g)%usb_params::D*step,step,step);

					cv::Mat d_patch(normalized_path,cv::Rect((si_index*step)%usb_params::D,(si_index/usb_params::g)%usb_params::D*step,step,step));

					//std::cout<<d_patch<<std::endl;

					calc_M_col(d_patch,si_index,i);
					
				}
			}
			catch(cv::Exception e)
			{
			//	std::cout<<e.err<<std::endl;
			}
		}
		

		M_bin_selection();

		split_M();
		//std::cout<<M<<std::endl;
		

	}
	void usb::asf_extraction()
	{
		for(int i=0;i<kps.size();i++)
		{
			int rp=usb_params::beta*kps[i].size;

			std::vector<float> query;
			query.push_back(kps[i].pt.x);
			query.push_back(kps[i].pt.y);
			std::vector<int> indices;
			std::vector<float> dists;
			indices.clear();
			dists.clear();
			cv::flann::SearchParams params(32);
			kps_kdtree->radiusSearch(query,indices,dists,rp,4,params);
		


			asf a;
			unsigned char* p=M.data;
			int step=M.cols;

			for(int j=0;j<indices.size();j++)
			{
				if(!dists[j])
					continue;

				int v=0;
				int col_index=indices[j];

				cv::Mat v_mat(8,1,CV_8U);
				unsigned char* p_vm=v_mat.data;
				for(int m=0;m<8;m++)
					p[m*1+0]+=p[m*step+col_index];

				

				int ori=abs(kps[i].angle-kps[col_index].angle)*usb_params::m/(720.0);

				int dis=sqrt((kps[i].pt.x-kps[col_index].pt.x)*(kps[i].pt.x-kps[col_index].pt.x)+(kps[i].pt.y-kps[col_index].pt.y)*(kps[i].pt.y-kps[col_index].pt.y)*usb_params::m/rp);
				a.push_back(col_index,v_mat,ori,dis);
			}
			asfs.push_back(a);
		}
		
	}
	void usb::calc_M_col(const cv::Mat& sub_img,int si_index,int col_index)
	{
		int sum=0;
		unsigned char* p_si=sub_img.data;
		unsigned char* p_M=M.data;
		int step=sub_img.cols;

		
		for(int i=0;i<5;i++)
		{
			bool res=calc_haar(sub_img,i);
			p_M[(si_index*5+i)*M.cols+col_index]=res;
			//M.at<int>(si_index*5+i,col_index)=res;
		}
	}

	bool usb::calc_haar(const cv::Mat& sub_img,int i)
	{
		unsigned char* p=sub_img.data;
		int a=usb_params::D/usb_params::g;
		int sum=0;
		int step=sub_img.cols;

		switch (i)
		{
		case 0:
			for(int i=0;i<a;i++)
				for(int j=0;j<a;j++)
					sum+=p[i*step+j]*(j>=a/2?1:-1);
			break;
		case 1:
			for(int i=0;i<a;i++)
				for(int j=0;j<a;j++)
					sum+=p[i*step+j]*(i>=a/2?1:-1);
			break;
		case 2:
			for(int i=0;i<a;i++)
				for(int j=0;j<a;j++)
					sum+=p[i*step+j]*((i-a/2)*(j-a/2)>=0?-1:1);
			break;
		case 3:
			for(int i=0;i<a;i++)
				for(int j=0;j<a;j++)
					sum+=p[i*step+j]*(j>=a/4&&j<a*3/4?-1:1);
			break;
		case 4:
			for(int i=0;i<a;i++)
				for(int j=0;j<a;j++)
					sum+=p[i*step+j]*(i>=a/4&&i<a*3/4?-1:1);
			break;
		default:
			break;
		}
		
		return sum>0;
	}

	void usb::M_bin_selection()
	{
		

		cv::Mat dst;
		std::set<int> P1;
		std::set<int> P2;

		unsigned char* p=M.data;
		int step=M.cols;

		float m_MD=MD(M.rows-1);

		int init_i=0;
		for(int i=0;i<M.rows;i++)
		{
			float t=MD(i);
			if(m_MD>t)
			{
				m_MD=t;
				init_i=i;
			}
		}
		P1.insert(init_i);

		for(int i=0;i<M.rows;i++)
			if(i!=init_i)
				P2.insert(i);

		while(P1.size()<usb_params::d && P2.size()>0)
		{
			std::set<int>::iterator it;
			int sel_i=*P2.begin();
			float m_MD=MD(*P2.begin());

			for(it=P2.begin();it!=P2.end();it++)
			{
				float t=MD(*it);
				if(m_MD>t)
				{
					sel_i=*it;
					m_MD=t;
				}
			}
			
			int swch=0;
			for(it=P1.begin();it!=P1.end();it++)
			{
				float res=CR(sel_i,*it);

				if(res>=usb_params::omega)
				{
					swch=1;
					break;
				}
			}
			
			if(swch && P2.size()>=usb_params::d)
				P2.erase(sel_i);
			else
			{
				P2.erase(sel_i);
				P1.insert(sel_i);	
			}
		}

		cv::Mat M_t(usb_params::d,kps.size(),CV_8U);
		unsigned char* p_t=M_t.data;

		std::set<int>::iterator it;
		int row_index=0;

		

		for(it=P1.begin();it!=P1.end();it++)
		{
			for(int i=0;i<step;i++)
				p_t[row_index*step+i]=p[(*it)*step+i];
			row_index++;
		}
		
	//	std::cout<<M<<std::endl;
		//std::cout<<M_t<<std::endl;



		M=M_t;
		M=M_t.clone();
		//std::cout<<M<<std::endl;

		
		/*p=M.data;

		for(int i=0;i<M.rows;i++)
			for(int j=0;j<M.cols;j++)
				std::cout<<(int)p[i*step+j]<<std::endl;*/
	}
	void usb::split_M()
	{
		usbs=std::vector<cv::Mat>(usb_params::n);
		unsigned char* p_M_col;
		unsigned char* p_M=M.data;
		int step=M.cols;
		
		for(int i=0;i<usb_params::n;i++)
		{
			cv::Mat M_col(M.rows,1,CV_8U);
			p_M_col=M_col.data;
			for(int j=0;j<M.rows;j++)
				p_M_col[j]=p_M[j*step+i];
			
			usbs[i]=M_col.clone();
		}
	}

	float usb::MD(int i)
	{
		unsigned char* p=M.data;
		int step=M.cols;
		float mean=0;
		for(int j=0;j<step;j++)
			mean+=p[i*step+j];
		mean/=step;
		return abs(mean-0.5);
	}
	float usb::CR(int i,int j)
	{
		unsigned char* p=M.data;
		int step=M.cols;

		float a=cv::normHamming(p+i*step,p+j*step,step,1);
		float b=step-cv::normHamming(p+i*step,p+j*step,step,1);
		
		return a>(step-b)?a/step:b/step;
	}


	void usb::show_kps(const char* win_name)
	{
		cv::Mat canvas;
		cv::drawKeypoints(src,kps,canvas);
		cv::imshow(win_name,canvas);
		cv::waitKey(0);

	}

	int usb::kp_match(const cv::Mat& Ua,const cv::Mat& Ub,const asf& Sa,const asf& Sb)
	{
		static int num=0;

		int match_order=0;

		//std::cout<<"a:"<<std::endl<<Ua<<std::endl;
		//std::cout<<"b:"<<std::endl<<Ub<<std::endl;

		float hamming_dis=cv::normHamming(Ua.data,Ub.data,Ua.cols*Ua.rows,1);
		//std::cout<<Ua<<std::endl;
		//std::cout<<Ub<<std::endl;

		/*
		if(hamming_dis<=5)
		{
			num++;
			std::cout<<num<<std::endl;
		}
		*/

		if(hamming_dis>usb_params::phi)
			return -1;
		
		for(int i=0;i<Sa.vals.size();i++)
			for(int j=0;j<Sb.vals.size();j++)
			{
				float ham_dis=cv::normHamming(Sa.vals[i].v.data,Sb.vals[j].v.data,8,1);
				if(ham_dis>usb_params::gama)
					continue;

				float ori_dis=MIN(usb_params::m-abs(Sa.vals[i].ori-Sb.vals[j].ori),abs(Sa.vals[i].ori-Sb.vals[j].ori));
				if(ori_dis>usb_params::eta)
					continue;

				float eculi_dis=abs(Sa.vals[i].dis-Sb.vals[j].dis);
				if(!(eculi_dis>usb_params::mu))
					match_order++;
			}

		return match_order;

	}

	int match(const usb& a, const usb& b, std::vector<int>& res)
	{
		int num=0;
		res=std::vector<int>(a.kps.size());
		
		/*int **order=new int*[a.kps.size()];
		for(int i=0;i<a.kps.size();i++)
			order[i]=new int[b.kps.size()];
*/
		std::vector<int> res_order;
		res_order.assign(a.kps.size(),-1);// initiate all the elem with -1

		for(int i=0;i<res.size();i++)
			res[i]=-1;

		for(int i=0;i<a.kps.size();i++)
			for(int j=0;j<b.kps.size();j++)
			{
				int match_order=usb::kp_match(a.usbs[i],b.usbs[j],a.asfs[i],b.asfs[j]);
				//if(match_order>=usb_params::match_order)
				if(match_order>res_order[i])
				{
					res_order[i]=match_order;
					res[i]=j;
				}
				
			}

		for(int i=0;i<res.size();i++)
			if(res[i]!=-1)
				num++;
		return num;
	}
		
}
