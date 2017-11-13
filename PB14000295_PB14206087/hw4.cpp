#include "FindContours.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;

#define black 255
#define white 0


int find_next_point(uchar* data, int *x, int *y, int cols, vector<cv::Point> &a,int *detect_point_offset)
{
	int forward_x[8] = { -1,-1,-1,0,1,1,1,0 };
	int forward_y[8] = { -1,0,1,1,1,0,-1,-1 };
	if (NULL == data)
	{
		cout << "Input Data Empty!" << endl;
	}
	int index = 0;
	for (; index < 8; index++)
	{
		int index1 = *(detect_point_offset + index);
		int x_new = *(forward_x + index1)+*x;
		int y_new = *(forward_y + index1)+*y;
		if (*(data + x_new*cols + y_new) == black)
		{}
		else
		{
			a.push_back(Point(y_new-1, x_new-1));
			*x = x_new;
			*y = y_new;
			break;
		}
	}
	int j;
	switch (index)
	{
		case 8: j = 8; break;
		default: j = detect_point_offset[index]; break;
	}
	return j;
}

int USTC_Find_Contours(Mat binaryImg, vector<vector<cv::Point>>&contours)
{
	if (NULL == binaryImg.data)
	{
		cout << "invaild input!" << endl;
		return MY_FAIL;
	}
	int rows = binaryImg.rows;
	int cols = binaryImg.cols;
	int cols_extend = cols + 2;
	int rows_extend = rows + 2;
	int circle_for_seven_to_zero[16] = { 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7 };
	//in order to get 0 after 7 just using +1 of the address
	int k[8] = { 4,5,6,7,0,1,2,3 };
	//A1 point is the last point of A2,then when A2 is j according to A1,and A1 will be k[j] according to A2
	uchar* data1 = binaryImg.data;
	Mat binary_extend(rows_extend, cols_extend, CV_8UC1, Scalar(black));
	uchar* data = binary_extend.data;
	for (int x = 0; x < rows; x++)
	{
		for (int y = 0; y < cols; y++)
		{
			int t1 = x*cols + y;
			int t2 = (x + 1)*(cols_extend)+y + 1;
			data[t2] = data1[t1];
		}
	}
	//in order to extend the image preventing the invaild address when the forward_x or forward_y=-1
	for (int origin_x = 1; origin_x < rows + 1; origin_x++)
	{
		for (int origin_y = 1; origin_y < cols + 1; origin_y++)
		{
			if (data[origin_x*cols_extend + origin_y] != black)
			{
				vector<cv::Point> a;
				a.push_back(Point(origin_y-1, origin_x-1));
				int j = 3;
				int x_new_1 = origin_x;
				int y_new_1 = origin_y;
				int detect_point_offset[8] = { 0 };
				int index2 = 0;
				while (1)
				{
					for (int col_i = 0; col_i < 8; col_i++)
					{
						detect_point_offset[col_i] = circle_for_seven_to_zero[index2];
						index2++;
					}
					int j = find_next_point(data, &x_new_1, &y_new_1, cols_extend, a,detect_point_offset);
					if (j == 8)
					{
						data[origin_x*cols_extend + origin_y] = black;
						a.clear();
						break;
					}
					else
					{
						index2 = k[j] + 1;
					}
					if (a.back() == Point(origin_y-1, origin_x-1))
					{
						break;
					}
				}
				if (a.size() == NULL)
				{}
				else
				{
					contours.push_back(a);
					drawContours(binary_extend, contours, contours.size()-1, Scalar(black),-1);	
					vector <cv::Point>().swap(a);
				}
			}
		}
	}

	Mat test_img(rows, cols, CV_8UC1, Scalar(black));
	drawContours(test_img, contours, -1, Scalar(white), 2);
	imshow("test", test_img);
	imshow("raw", binaryImg);
	waitKey();
}
