#include "FindContours.h"


int circle_by_index_r[8][7] = { { 2,1,0,7,6,3,5 },{ 3,2,1,0,7,4,6 },{ 4,3,2,1,0,5,7 },{ 5,4,3,2,1,6,0 },{ 6,5,4,3,2,7,1 },{ 7,6,5,4,3,2,0 },{ 0,7,6,5,4,3,1 },{ 1,0,7,6,5,4,2 } };

int circle_by_index_l[8][7] = { { 6,7,0,1,2,5,3 },{ 7,0,1,2,3,6,4 },{ 0,1,2,3,4,7,5 },{ 1,2,3,4,5,0,6 },{ 2,3,4,5,6,1,7 },{ 3,4,5,6,7,0,2 },{ 4,5,6,7,0,1,3 },{ 5,6,7,0,1,2,4 } };

int USTC_Find_Contours(Mat binaryImg, vector < vector < cv::Point >>& contours)

{

	if (NULL == binaryImg.data)

	{

		cout << "img imput erro";

		return MY_FAIL;

	}



	int width = binaryImg.cols;

	int height = binaryImg.rows;



	Mat labelImg(height, width, CV_32SC1);

	int* T = NULL;

	int size_T = ((width*height) >> 2) + 1;

	T = new int[size_T];



	if (NULL == T)

	{

		cout << "no enough tip";

		return MY_FAIL;

	}



	memset(T, 0, size_T * sizeof(int));

	int width_ex = width + 2;

	int height_ex = height + 2;

	Mat ex_binaryImg(height_ex, width_ex, CV_8UC1, Scalar(0));

	Mat labelImg2(height_ex, width_ex, CV_32SC1, Scalar(0));







	Mat imageROI = ex_binaryImg(Rect(1, 1, width, height));

	binaryImg.copyTo(imageROI);



	uchar *ptr_ex_bin = ex_binaryImg.data;

	int *ptr_ex_lab = (int*)labelImg2.data;



	int maxlabel = 1;

	int height_of_initial = height + 1;

	int width_of_initial = width + 1;



	for (int i = 1; i < height_of_initial; i++)

	{



		int temp_x2 = i * width_ex;

		int temp_x1 = temp_x2 - width_ex;

		for (int j = 1; j < width_of_initial; j++)

		{

			int loc = temp_x2 + j;



			if (!ptr_ex_bin[loc])

			{

				continue;

			}





			int loc_b = loc - width_ex;

			int loc_a = loc_b - 1;

			int loc_c = loc_b + 1;

			int loc_d = loc - 1;



			int a = ptr_ex_bin[loc_a];

			int b = ptr_ex_bin[loc_b];

			int c = ptr_ex_bin[loc_c];

			int d = ptr_ex_bin[loc_d];

			int label_a = ptr_ex_lab[loc_a];

			int label_b = ptr_ex_lab[loc_b];

			int label_c = ptr_ex_lab[loc_c];

			int label_d = ptr_ex_lab[loc_d];



			int bi_label = (a << 3) + (b << 2) + (c << 1) + d;

			int min;

			int t_label_a;

			int t_label_b;

			int t_label_c;

			int t_label_d;

			switch (bi_label)

			{

			case 0://0000

				ptr_ex_lab[loc] = maxlabel;

				T[maxlabel] = maxlabel;

				maxlabel++;

				break;

			case 255://0001

				ptr_ex_lab[loc] = T[label_d];

				break;

			case 510://0010

				ptr_ex_lab[loc] = T[label_c];

				break;

			case 765://0011

				t_label_c = T[label_c];

				t_label_d = T[label_d];

				if (t_label_c <= t_label_d)

				{

					ptr_ex_lab[loc] = t_label_c;

					T[label_d] = t_label_c;

				}

				else

				{

					ptr_ex_lab[loc] = t_label_d;

					T[label_c] = t_label_d;

				}

				break;

			case 1020://0100

				ptr_ex_lab[loc] = T[label_b];

				break;

			case 1275://0101

				t_label_b = T[label_b];

				t_label_d = T[label_d];

				if (t_label_b <= t_label_d)

				{

					ptr_ex_lab[loc] = t_label_b;

					T[label_d] = t_label_b;

				}

				else

				{

					ptr_ex_lab[loc] = t_label_d;

					T[label_b] = t_label_d;

				}

				break;

			case 1530://0110

				t_label_b = T[label_b];

				t_label_c = T[label_c];

				if (t_label_b <= t_label_c)

				{

					ptr_ex_lab[loc] = t_label_b;

					T[label_c] = t_label_b;

				}

				else

				{

					ptr_ex_lab[loc] = t_label_c;

					T[label_b] = t_label_c;

				}

				break;

			case 1785://0111

				min = T[label_b];

				if (min > T[label_c])

					min = T[label_c];

				if (min > T[label_d])

					min = T[label_d];

				ptr_ex_lab[loc] = min;

				T[label_b] = min;

				T[label_c] = min;

				T[label_d] = min;

				break;

			case 2040://1000

				ptr_ex_lab[loc] = T[label_a];

				break;

			case 2295://1001

				t_label_a = T[label_a];

				t_label_d = T[label_d];

				if (t_label_a <= t_label_d)

				{

					ptr_ex_lab[loc] = t_label_a;

					T[label_d] = t_label_a;

				}

				else

				{

					ptr_ex_lab[loc] = t_label_d;

					T[label_a] = t_label_d;

				}

				break;

			case 2550://1010

				t_label_a = T[label_a];

				t_label_c = T[label_c];

				if (t_label_a <= t_label_c)

				{

					ptr_ex_lab[loc] = t_label_a;

					T[label_c] = t_label_a;

				}

				else

				{

					ptr_ex_lab[loc] = t_label_c;

					T[label_a] = t_label_c;

				}

				break;

			case 2805://1011

				min = T[label_a];

				if (min > T[label_c])

					min = T[label_c];

				if (min > T[label_d])

					min = T[label_d];

				ptr_ex_lab[loc] = min;

				T[label_a] = min;

				T[label_c] = min;

				T[label_d] = min;

				break;

			case 3060://1100

				t_label_a = T[label_a];

				t_label_b = T[label_b];

				if (t_label_a <= t_label_b)

				{

					ptr_ex_lab[loc] = t_label_a;

					T[label_b] = t_label_a;

				}

				else

				{

					ptr_ex_lab[loc] = t_label_b;

					T[label_a] = t_label_b;

				}

				break;

			case 3315://1101

				min = T[label_a];

				if (min > T[label_b])

					min = T[label_b];

				if (min > T[label_d])

					min = T[label_d];

				ptr_ex_lab[loc] = min;

				T[label_a] = min;

				T[label_b] = min;

				T[label_d] = min;

				break;

			case 3570://1110

				min = T[label_a];

				if (min > T[label_b])

					min = T[label_b];

				if (min > T[label_c])

					min = T[label_c];

				ptr_ex_lab[loc] = min;

				T[label_a] = min;

				T[label_b] = min;

				T[label_c] = min;

				break;

			case 3825://1111

				min = T[label_a];

				if (min > T[label_b])

					min = T[label_b];

				if (min > T[label_c])

					min = T[label_c];

				if (min > T[label_d])

					min = T[label_d];

				ptr_ex_lab[loc] = min;

				T[label_a] = min;

				T[label_b] = min;

				T[label_c] = min;

				T[label_d] = min;

				break;





			}

		}

	}



	for (int i = 1; i < height_of_initial; i++)

	{

		int temp_loc2 = i*width_ex;

		for (int j = 1; j < width_of_initial; j++)

		{

			int loc2 = temp_loc2 + j;

			ptr_ex_lab[loc2] = T[ptr_ex_lab[loc2]];

		}

	}



	delete[] T;





	contours.clear();

	int curr_label;

	vector<cv::Point>curr_contour;

	int area = height_of_initial*width_ex;

	int label_curr = 0;

	int* labelImg2_fast = (int*)(labelImg2.data);

	for (int index = width; index < area; index++)

	{



		if (labelImg2_fast[index] <= label_curr)

		{

			continue;

		}



		label_curr = labelImg2_fast[index];

		curr_contour.clear();

		int y = index / width_ex;

		int x = index % width_ex;

		curr_contour.push_back(cv::Point(x - 1, y - 1));

		curr_label = labelImg2_fast[index];

		cv::Point start_pt = cv::Point(x, y);

		cv::Point last_pt;

		cv::Point curr_pt = start_pt;

		cv::Point next_pt(-1, -1);



		int start_index = 0;

		int end_index = 7;

		int index_for_next_pt = -1;



		for (int neighbour_index = start_index; neighbour_index <= end_index; neighbour_index++)

		{

			cv::Point neighbourPt;



			switch (neighbour_index)

			{

			case 0:neighbourPt = Point(curr_pt.x + 1, curr_pt.y); break;

			case 1:neighbourPt = Point(curr_pt.x + 1, curr_pt.y - 1); break;

			case 2:neighbourPt = Point(curr_pt.x, curr_pt.y - 1); break;

			case 3:neighbourPt = Point(curr_pt.x - 1, curr_pt.y - 1); break;

			case 4:neighbourPt = Point(curr_pt.x - 1, curr_pt.y); break;

			case 5:neighbourPt = Point(curr_pt.x - 1, curr_pt.y + 1); break;

			case 6:neighbourPt = Point(curr_pt.x, curr_pt.y + 1); break;

			case 7:neighbourPt = Point(curr_pt.x + 1, curr_pt.y + 1); break;

			}





			if (curr_label == labelImg2_fast[neighbourPt.y * width_ex + neighbourPt.x])

			{

				index_for_next_pt = neighbour_index;

				next_pt = neighbourPt;

				break;

			}

		}





		if (-1 == index_for_next_pt)

		{

			contours.push_back(curr_contour);

			continue;

		}

		else

		{

			cv::Point real_pt = cv::Point(next_pt.x - 1, next_pt.y - 1);

			curr_contour.push_back(real_pt);

		}



		int neighbour_index = 0;

		int temp_for_addition = -1;

		int(*circle_by_index)[7] = circle_by_index_r;

		while (next_pt != start_pt)

		{

			last_pt = curr_pt;

			curr_pt = next_pt;





			int flag = 0;

			int *round_f = circle_by_index[index_for_next_pt];

			for (int neighbour_index_f = 0; neighbour_index_f <7; neighbour_index_f++)

			{

				cv::Point neighbourPt;

				neighbour_index = round_f[neighbour_index_f];



				switch (neighbour_index)

				{

				case 0:neighbourPt = Point(curr_pt.x + 1, curr_pt.y); break;

				case 1:neighbourPt = Point(curr_pt.x + 1, curr_pt.y - 1); break;

				case 2:neighbourPt = Point(curr_pt.x, curr_pt.y - 1); break;

				case 3:neighbourPt = Point(curr_pt.x - 1, curr_pt.y - 1); break;

				case 4:neighbourPt = Point(curr_pt.x - 1, curr_pt.y); break;

				case 5:neighbourPt = Point(curr_pt.x - 1, curr_pt.y + 1); break;

				case 6:neighbourPt = Point(curr_pt.x, curr_pt.y + 1); break;

				case 7:neighbourPt = Point(curr_pt.x + 1, curr_pt.y + 1); break;

				}





				if (curr_label == labelImg2_fast[neighbourPt.y * width_ex + neighbourPt.x])

				{

					index_for_next_pt = neighbour_index;

					next_pt = neighbourPt;

					flag = 1;

					cv::Point real_pt = cv::Point(next_pt.x - 1, next_pt.y - 1);

					curr_contour.push_back(real_pt);

					break;

				}

			}

			if (!flag)

			{

				temp_for_addition++;





				if (1 == temp_for_addition)break;





				if (!temp_for_addition)

				{

					next_pt = Point(start_pt.x - 1, start_pt.y);

					index_for_next_pt = 4;

					circle_by_index = circle_by_index_l;

				}

			}

		}



		contours.push_back(curr_contour);



	}

	return MY_OK;

}
