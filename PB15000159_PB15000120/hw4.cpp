#include <opencv2/opencv.hpp>
#include <cv.h>
#include <vector>
#include <ctime>
#include <set>
#include <deque>

#define MAXN 5000
#define MY_OK 1
#define MY_FAIL -1

using namespace std;
using namespace cv;

int ustc_ConnectedComponentLabeling(
	Mat grayImg,
	Mat& labelImg,
	uchar bin_th)
{
	int i, j, k;
	int label;
	int x, y;
	int row = grayImg.rows, col = grayImg.cols;
	const int eight[8][2] = { { 0,-1 },{ -1,-1 },{ -1,0 },{ -1,1 },{ 0,1 },{ 1,-1 },{ 1,0 },{ 1,1 } };
	int temp, test;
	int ans[MAXN] = { 0 };
	static bool map[MAXN][MAXN];
	deque<int> queue;
	vector<vector<int> > after;
	queue.clear();
	static bool is[MAXN];
	bool flag = true;
	uchar *pg = NULL, *pg2;
	int *pl = NULL, *pl2;
	labelImg.create(Size(col, row), CV_32SC1);
	memset(is, false, MAXN);
	memset(map, false, MAXN * MAXN);
	label = 1;
	for (i = 0;i < row;i++)
	{
		pg2 = grayImg.ptr<uchar>(i);
		pl2 = labelImg.ptr<int>(i);
		if (i > 0)
		{
			pg = grayImg.ptr<uchar>(i - 1);
			pl = labelImg.ptr<int>(i - 1);
		}
		for (j = 0;j < col;j++)
		{
			if (pg2[j] < bin_th)
			{
				continue;
			}
			temp = label;
			x = i - 1;
			if (pl!=NULL)
			{
				//比较左上角标签值
				y = j - 1;
				if (y >= 0)
				{
					if (pg[y] >= bin_th)
					{
						temp = pl[y];
					}
				}
				//比较上方标签值 x=i-1
				y = j;
				if (pg[y] >= bin_th)
				{
					test = pl[y];
					if (temp != label)
					{
						if (!map[temp][test])
						{
							map[temp][test] = map[test][temp] = true;
						}
					}
					if (test < temp)
					{
						temp = test;
					}
				}
				//比较右上方标签值 x=i-1
				y = j + 1;
				if (y < col)
				{
					if (pg[y] >= bin_th)
					{
						test = pl[y];
						if (temp != label)
						{
							if (!map[temp][test])
							{
								map[temp][test] = map[test][temp] = true;
							}
						}
						if (test < temp)
						{
							temp = test;
						}
					}
				}
			}
			//比较左边标签值
			x = i;
			y = j - 1;
			if (y >= 0)
			{
				if (pg2[y] >= bin_th)
				{
					test = pl2[y];
					if (temp != label)
					{
						if (!map[temp][test])
						{
							map[temp][test] = map[test][temp] = true;
						}
					}
					if (test < temp)
					{
						temp = test;
					}
				}
			}

			if (temp == label)
			{
				label++;
				if (label > MAXN)
				{
					return MY_FAIL;
				}
			}
			pl2[j] = temp;
		}
	}
	after.resize(label + 1);
	for (i = 1;i < label;i++)
	{
		for (j = i+1;j < label;j++)
		{
			if (map[i][j])
			{
				after[i].push_back(j);
				after[j].push_back(i);
			}
		}
	}
	temp = 1;
	while (flag)
	{
		flag = false;
		for (i = 1;i < label;i++)
		{
			if (!is[i])
			{
				flag = true;
				break;
			}
		}
		queue.push_back(i);
		is[i] = true;
		while (!queue.empty())
		{
			test = queue.front();
			queue.pop_front();
			ans[test] = temp;
			for (j = 0;j < after[test].size();j++)
			{
				k = after[test][j];
				if (is[k])
				{
					continue;
				}
				queue.push_back(k);
				is[k] = true;
			}
		}
		temp++;
	}
/*	while (flag)
	{
		flag = false;
		for (i = 1;i < label;i++)
		{
			if (!is[i])
			{
				flag = true;
				break;
			}
		}
		//		queue.clear();
		queue.push_back(i);
		is[i] = true;
		while (!queue.empty())
		{
			test = queue.front();
			queue.pop_front();
			ans[test] = temp;
			for (i = 1;i < label;i++)
			{
				if (!map[test][i] || is[i])
				{
					continue;
				}
				queue.push_back(i);
				is[i] = true;
			}
		}
		temp++;
	}*/
	for (i = 0;i < row;i++)
	{
		pl = labelImg.ptr<int>(i);
		for (j = 0;j < col;j++)
		{
			pl[j] = ans[pl[j]];
		}
	}
	return MY_OK;
}

int USTC_Find_Contours(Mat binaryImg, vector < vector < cv::Point >>& contours)
{
	int i, j;
	int num = 0;
	int temp;
	int row = binaryImg.rows, col = binaryImg.cols;
	Mat label;
	int *ql;
	bool flag, end;
	ustc_ConnectedComponentLabeling(binaryImg, label, 128);
	contours.clear();
	contours.resize(MAXN);
	Point work, test;
	Point start[MAXN];
	bool is[MAXN];
	Point search[30] = { Point(0,1),Point(-1,1),Point(-1,0),Point(-1,-1),Point(0,-1),Point(1,-1),Point(1,0),Point(1,1),Point(0,1),Point(-1,1),Point(-1,0),Point(-1,-1),Point(0,-1),Point(1,-1),Point(1,0),Point(1,1), Point(0,1),Point(-1,1),Point(-1,0),Point(-1,-1),Point(0,-1),Point(1,-1),Point(1,0),Point(1,1) };
	memset(is, false, MAXN);
	for (i = 0;i < row;i++)
	{
		ql = label.ptr<int>(i);
		for (j = 0;j < col;j++)
		{
			if (ql[j] && !is[ql[j]])
			{
				start[ql[j]] = Point(i, j);
				is[ql[j]] = true;
				num++;
			}
		}
	}
	for (i = 1;i <= num;i++)
	{
		temp = 0;
		flag = true;
		if (is[i])
		{
			work = start[i];
			contours[i].push_back(work);
			end = false;
			while (flag)
			{
				flag = false;
				temp += 11;
				for (j = temp - 5;j < temp;j++)
				{
					test = work + search[j];
					if (test.x >= 0 && test.x < row && test.y >= 0 && test.y < col)
					{
						if (label.at<int>(test.x,test.y) == i)
						{
							if (test == start[i])
							{
								end = true;
								break;
							}
							flag = true;
							work = test;
							temp = j % 8;
							contours[i].push_back(test);
							break;
						}
					}
				}
				if (!flag && !end)
				{
//					cout << test << " " << label.at<int>(test) << endl;
					test = work + search[temp];
					if (test.x >= 0 && test.x < row&&test.y >= 0 && test.y < col)
					{
						if (label.at<int>(test.x,test.y) == i)
						{
							if (test == start[i])
							{
								end = true;
								continue;
							}
							flag = true;
							work = test;
							temp = j % 8;
							contours[i].push_back(test);
							continue;
						}
					}
//					cout << test << " " << label.at<int>(test) << endl;
					test = work + search[temp + 2];
					if (test.x >= 0 && test.x < row&&test.y >= 0 && test.y < col)
					{
						if (label.at<int>(test.x,test.y) == i)
						{
							if (test == start[i])
							{
								end = true;
								continue;
							}
							flag = true;
							work = test;
							temp = j % 8;
							contours[i].push_back(test);
							continue;
						}
					}
				}
			}
		}
	}
	return MY_OK;
}
