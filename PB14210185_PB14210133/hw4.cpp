#include "FindContours.h"

int USTC_Find_Contours(Mat binaryImg, vector< vector< cv::Point >>& contours)
{
	if (NULL == binaryImg.data || CV_8UC1 != binaryImg.type()) 
	{
		cout << "Input image is invalid." << endl;
		return MY_FAIL;
	}

	int height = binaryImg.rows, width = binaryImg.cols;
	Mat ex_binaryImg(height + 2, width + 2, CV_8UC1, Scalar(0));
	for (int i = 0; i < height; i++) {
		uint8_t *grayImg_line = binaryImg.ptr<uint8_t>(i);
		uint8_t *binaryImg_line = ex_binaryImg.ptr<uint8_t>(i + 1);
		for (int j = 0; j < width; j++) {
			binaryImg_line[j + 1] = ((100 - grayImg_line[j]) >> 31) & 1;
		}
	}

	Mat ex_labelImg(height + 2, width + 2, CV_32SC1, Scalar(0));
	vector<int> result;
	result.push_back(0);
	for (int i = 1; i <= height; i++) 
	{
		uchar *binImg_preline = ex_binaryImg.ptr<uchar>(i - 1);
		uchar *binImg_line = ex_binaryImg.ptr<uchar>(i);
		long *labImg_preline = ex_labelImg.ptr<long>(i - 1);
		long *labImg_line = ex_labelImg.ptr<long>(i);
		for (int j = 1; j <= width; j++) 
		{
			if (binImg_line[j]) 
			{
				int num = 0;
				int nlabels[4];
				if (binImg_preline[j - 1]) 
				{
					nlabels[num] = labImg_preline[j - 1];
					num++;
				}
				if (binImg_preline[j]) 
				{
					nlabels[num] = labImg_preline[j];
					num++;
				}
				if (binImg_preline[j + 1]) 
				{
					nlabels[num] = labImg_preline[j + 1];
					num++;
				}
				if (binImg_line[j - 1]) 
				{
					nlabels[num] = labImg_line[j - 1];
					num++;
				}
				if (0 == num) 
				{
					labImg_line[j] = (int)result.size();
					result.push_back(0);
				}
				else 
				{
					int min = nlabels[0];
					for (int k = 1; k < num; k++) 
					{
						if (nlabels[k] < min) 
						{
							min = nlabels[k];
						}
					}
					labImg_line[j] = min;
					for (int k = 0; k < num; k++) 
					{
						if (nlabels[k] != min) 
						{
							int big = nlabels[k];
							int small = min;
							while (small != result.at(big)) 
							{
								if (0 == result.at(big)) 
								{
									result.at(big) = small;
									break;
								}
								else 
								{
									big = result.at(big);
									if (big < small) 
									{
										int temp = big;
										big = small;
										small = temp;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	size_t length = result.size();
	for (int i = 1; i < length; i++) {
		if (0 == result.at(i)) {
			result.at(i) = i;
		}
		else {
			result.at(i) = result.at(result.at(i));
		}
	}
	Mat labelImg(height + 2, width + 2, CV_32SC1, Scalar(0));
	for (int i = 1; i <= height; i++) 
	{
		long *labImg_line = labelImg.ptr<long>(i);
		long *exlabImg_line = ex_labelImg.ptr<long>(i);
		for (int j = 1; j <= width; j++) 
		{
			labImg_line[j] = result.at(exlabImg_line[j]);
		}
	}

	int row_bias[8];
	int col_bias[8];
	row_bias[0] = 0; col_bias[0] = 1;
	row_bias[1] = -1; col_bias[1] = 1;
	row_bias[2] = -1; col_bias[2] = 0;
	row_bias[3] = -1; col_bias[3] = -1;
	row_bias[4] = 0; col_bias[4] = -1;
	row_bias[5] = 1; col_bias[5] = -1;
	row_bias[6] = 1; col_bias[6] = 0;
	row_bias[7] = 1; col_bias[7] = 1;

	vector<long> used_label;
	used_label.push_back(0);

	contours.clear();
	long* ptr_data = (long*)labelImg.data;
	for (int i = 1; i <= height; i++)
	{
		int i_width = i*(width + 2);
		for (int j = 1; j <= width; j++)
		{
			if (used_label.end() == find(used_label.begin(), used_label.end(), ptr_data[i_width + j]))
			{
				long label = ptr_data[i_width + j];
				Point first_posi = Point(j, i);
				Point now_posi = first_posi;
				int last_posi;
				vector<Point> contour;
				contour.push_back(Point(j - 1, i - 1));

				for (int p = 5; p <= 8; p++)
				{
					int posi = p % 8;
					int row = i + row_bias[posi];
					int col = j + col_bias[posi];
					if (ptr_data[row*(width + 2) + col] == label)
					{
						last_posi = (posi + 4) % 8;
						now_posi = Point(col, row);
						break;
					}
				}

				if (now_posi == first_posi)
				{
					contour.push_back(Point(first_posi.x - 1, first_posi.y - 1));
					contours.push_back(contour);
					continue;
				}
				
				Point second_posi = now_posi;

				while(1)
				{
					contour.push_back(Point(now_posi.x - 1, now_posi.y - 1));
					bool end_flag = true;
					for (int p = 2; p <= 6; p++)
					{
						int posi = (last_posi + p) % 8;
						int row = now_posi.y + row_bias[posi];
						int col = now_posi.x + col_bias[posi];
						if (ptr_data[row*(width + 2) + col] == label)
						{
							last_posi = (posi + 4) % 8;
							now_posi = Point(col, row);
							end_flag = false;
							break;
						}
					}
					if (end_flag)
					{
						now_posi = Point(now_posi.x + col_bias[last_posi], now_posi.y + row_bias[last_posi]);
						last_posi = (last_posi + 4) % 8;
					}
					if (now_posi == first_posi) {
						contour.push_back(Point(now_posi.x - 1, now_posi.y - 1));
						bool end_flag = true;
						for (int p = 2; p <= 6; p++)
						{
							int posi = (last_posi + p) % 8;
							int row = now_posi.y + row_bias[posi];
							int col = now_posi.x + col_bias[posi];
							if (ptr_data[row*(width + 2) + col] == label)
							{
								last_posi = (posi + 4) % 8;
								now_posi = Point(col, row);
								end_flag = false;
								break;
							}
						}
						if (end_flag || now_posi == second_posi)
						{
							break;
						}
					}
				}
				contours.push_back(contour);
				used_label.push_back(label);
			}
		}
	}

	return MY_OK;
}
