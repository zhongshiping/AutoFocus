#include "opencv2/opencv.hpp"
#include "iomanip"

using namespace cv;
using namespace std;

unsigned long Tenengrad_measure(Mat& image)
{
  unsigned long score = 0;
  for (int x = 0; x < image.rows; ++x)
  {
    uchar* ptr = image.ptr<uchar>(x);
#pragma omp parallel for
    for (int y = 0; y < image.cols*image.channels(); ++y)
    {
      score += ptr[y];
    }
  }
  return score;
}

double Tenengrad(Mat& roi)
{
  Mat smooth_image;
  blur(roi, smooth_image, Size(3, 3));
  Mat sobel_x_mat, sobel_y_mat;
  Sobel(smooth_image, sobel_x_mat, -1, 1, 0, 0);
  Sobel(smooth_image, sobel_y_mat, -1, 0, 1, 3);

  convertScaleAbs(sobel_x_mat, sobel_x_mat);
  convertScaleAbs(sobel_y_mat, sobel_y_mat);

  Mat pow_x;
  Mat pow_y;
  pow(sobel_x_mat, 2, pow_x);
  pow(sobel_y_mat, 2, pow_y);

  Mat weighted_mat;
  addWeighted(pow_x, 1, pow_y, 1, 0, weighted_mat);

  double score = Tenengrad_measure(weighted_mat);

  return score;
}

double Brenner(Mat& image)
{
  double score = 0.0;
  for (int i = 0; i < image.rows; ++i)
  {
    uchar *ptr = image.ptr<uchar>(i);
#pragma omp parallel for
    for (int j = 0; j < (image.cols - 2)*image.channels(); ++j)
    {
      score += (ptr[j + 2] - ptr[j])*(ptr[j + 2] - ptr[j]);
    }
  }
  return score;
}

double SMD2(Mat& image)
{
  unsigned long score = 0;
  for (int i = 0; i < image.rows - 1; ++i)
  {
    uchar* ptr = image.ptr<uchar>(i);
    uchar* ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
    for (int j = 0; j < (image.cols - 1)*image.channels(); ++j)
    {
      score += (ptr[j] - ptr[j + 1]) * (ptr[j] - ptr_1[j]);
    }
  }
  return score;
}

double Energy(Mat& image)
{
  unsigned long score = 0;
  for (int i = 0; i < image.rows - 1; ++i)
  {
    uchar* ptr = image.ptr<uchar>(i);
    uchar* ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
    for (int j = 0; j < (image.cols - 1) * image.channels(); ++j)
    {
      score += pow(ptr[j + 1] - ptr[j], 2) + pow(ptr_1[j] - ptr[j], 2);
    }
  }
  return score;
}

double Jpeg(Mat& image)
{
  //horizontal calculate
  auto b_h = 0.0;
  for (int i = 1; i < floor(image.rows / 8) - 1; ++i)
  {
    uchar* ptr = image.ptr<uchar>(8 * i);
    uchar* ptr_1 = image.ptr<uchar>(8 * i + 1);
#pragma omp parallel for
    for (int j = 1; j < image.cols; ++j)
    {
      b_h += abs(ptr_1[j] - ptr[j]);
    }
  }
  b_h *= 1 / (image.cols*(floor(image.rows / 8) - 1));

  auto a_h = 0.0;
  for (int i = 1; i < image.rows - 1; ++i)
  {
    uchar* ptr = image.ptr<uchar>(i);
    uchar* ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
    for (int j = 1; j < image.cols; ++j)
    {
      a_h += abs(ptr_1[j] - ptr[j]);
    }
  }
  a_h = (a_h * 8.0 / (image.cols * (image.rows - 1)) - b_h) / 7;

  auto z_h = 0.0;
  for (int i = 1; i < image.rows - 2; ++i)
  {
    uchar* ptr = image.ptr<uchar>(i);
    uchar* ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
    for (int j = 1; j < image.cols; ++j)
    {
      z_h += (ptr_1[j] - ptr[j]) * (ptr_1[j + 1] - ptr[j]) > 0 ? 0 : 1;
    }
  }
  z_h *= 1.0 / (image.cols* (image.rows - 2));

  //vertical calculate
  auto b_v = 0.0;
  for (int i = 1; i < image.rows; ++i)
  {
    uchar* ptr = image.ptr<uchar>(i);
#pragma omp parallel for
    for (int j = 1; j < floor(image.cols / 8) - 1; ++j)
    {
      b_v += abs(ptr[8 * j + 1] - ptr[8 * j]);
    }
  }
  b_v *= 1.0 / (image.rows*(floor(image.cols / 8) - 1));

  auto a_v = 0.0;
  for (int i = 1; i < image.rows; ++i)
  {
    uchar* ptr = image.ptr<uchar>(i);
#pragma omp parallel for
    for (int j = 1; j < image.cols - 1; ++j)
    {
      a_v += abs(ptr[j + 1] - ptr[j]);
    }
  }
  a_v = (a_v * 8.0 / (image.rows * (image.cols - 1)) - b_v) / 7;

  auto z_v = 0.0;
  for (int i = 1; i < image.rows; ++i)
  {
    uchar* ptr = image.ptr<uchar>(i);
#pragma omp parallel for
    for (int j = 1; j < image.cols - 2; ++j)
    {
      z_v += (ptr[j + 1] - ptr[j]) * (ptr[j + 2] - ptr[j + 1]) > 0 ? 0 : 1;
    }
  }
  z_v *= 1.0 / (image.rows* (image.cols - 2));

  ////////////////////////////////////////////////////////////////////////////
  auto B = (b_v + b_h) / 2;
  auto A = (a_h + a_v) / 2;
  auto Z = (z_h + z_v) / 2;
  auto alpha = -245.9, beta = 261.9, gamma1 = -0.024, gamma2 = 0.016, gamma3 = 0.0064;

  auto S = alpha + beta*pow(B, gamma1)*pow(A, gamma2)*pow(Z, gamma3);

  return S;
}

double Jpeg2(Mat& image)
{
  double s = Jpeg(image);
  double ss = 4.0 / (1.0 + exp(-1.0217*(s - 3))) + 1.0;
  return ss;
}

int main()
{
  VideoCapture capture;
  if (!capture.open("C://Users//sp_zh//Desktop//1.avi"))
  {
    return -1;
  }
  int frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
  int index = 0;
  vector<double> focuss;
  while (frames - index > 0)
  {
    Mat image;
    capture >> image;

    Rect roi_rect(1221, 409, 493, 1267);
    Mat roi = image(roi_rect);
    //double score = Tenengrad(roi);
    //double score = Brenner(roi);
    //double score = SMD2(roi);
    //double score = Energy(roi);
    //double score = Jpeg(roi);
    double score = Jpeg2(roi);
    char text[16];
    sprintf(text, "%lf", score);
    putText(image, text, Point(100, 100), FONT_HERSHEY_COMPLEX, 1.5, Scalar(255, 0, 0));
    imshow("1", image);
    waitKey(20);
    index++;
    focuss.push_back(score);
  }
  return 0;
}
