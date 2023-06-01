#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <opencv2/core/types_c.h>
#include <fstream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core_c.h>

using namespace cv;
using namespace std;

static IplImage* image = 0;
static IplImage* image_f = 0;
static IplImage* image2 = 0;

static const double M_PI = 3.141592;


#define IPL_DEPTH_8U 8

// параметры модели
double x_step;    // шаг модели по X
double y_step;    // шаг ммодели по Y
int n_x_steps;  // размерность модели по X
int n_y_steps;  // размерность модели по Y

// параметры SAR-съемки
double lengthOfWave;  // длинна волны
double azimuth;           // азимут
double inclination;           // наклон

// параметры интерферограммы
int first_day;   // день начала съемки
int second_day;  //день окочания съемки
int XRE_SUMPL;  // отношение шага модели к шагу съекки
int YRE_SUMPL;  // отношение шага модели к шагу съекки
double xf_step;    // шаг съемки по X
double yf_step;    // шаг съемки по Y
int n_xf_steps;  // размерность по X
int n_yf_steps;  // размерность по Y

// переменные и массивы
float* xy_array1;  // оседания по модели
float* xy_array2;  // оседания на узлах интерферограммы


static double bi_val_1(double xbi, double ybi, bool ascending) {
  int x_ind = xbi / x_step;
  int y_ind = ybi / y_step;
  double x_loc = xbi / x_step - x_ind;
  double y_loc = ybi / y_step - y_ind;
  double bi_val = ((1.0 - x_loc) * xy_array1[x_ind * n_y_steps + y_ind] +
                   x_loc * xy_array1[(x_ind + 1) * n_y_steps + y_ind]) *
                      (1.0 - y_loc) +
                  ((1.0 - x_loc) * xy_array1[x_ind * n_y_steps + y_ind + 1] +
                   x_loc * xy_array1[(x_ind + 1) * n_y_steps + y_ind + 1]) *
                      y_loc;

  return bi_val;
}

static double bi_int(int x_ind, int y_ind, bool ascending) {
  double x_loc = azimuth;
  double loc_inclination = inclination;
  if (!ascending) {
    x_loc = M_PI - azimuth;
    loc_inclination = M_PI - inclination;
  }
  double y_loc = sin(loc_inclination);
  double xd_loc = x_loc / 2.0;
  double yd_loc = y_loc / 2.0;
  double l_loc = sqrt(x_loc * x_loc * xf_step * xf_step +
                    y_loc * y_loc * yf_step * yf_step);
  double lod_val = 0.0;
  lod_val = loc_inclination * l_loc;
  double bi_val = ((1.0 - x_loc) * xy_array2[x_ind * n_yf_steps + y_ind] +
                 x_loc * xy_array2[(x_ind + 1) * n_yf_steps + y_ind]) *
                    (1.0 - y_loc) +
                ((1.0 - x_loc) * xy_array2[x_ind * n_yf_steps + y_ind + 1] +
                 x_loc * xy_array2[(x_ind + 1) * n_yf_steps + y_ind + 1]) *
                    y_loc;

  for (int k = 0; k < 10; k++) {  // Итерация бисекции
      lod_val = loc_inclination * l_loc;
    bi_val = ((1.0 - x_loc) * xy_array2[x_ind * n_yf_steps + y_ind] +
              x_loc * xy_array2[(x_ind + 1) * n_yf_steps + y_ind]) *
                 (1.0 - y_loc) +
             ((1.0 - x_loc) * xy_array2[x_ind * n_yf_steps + y_ind + 1] +
              x_loc * xy_array2[(x_ind + 1) * n_yf_steps + y_ind + 1]) *
                 y_loc;
    if (lod_val > bi_val) {
      x_loc -= xd_loc;
      y_loc -= yd_loc;
      l_loc *= 0.5;
    } else {
      x_loc += xd_loc;
      y_loc += yd_loc;
      l_loc *= 1.5;
    }
    xd_loc *= 0.5;
    yd_loc *= 0.5;
  }
  return (sqrt(l_loc * l_loc + bi_val * bi_val) +
          ((double)(x_ind + y_ind) / (double)(n_yf_steps + n_xf_steps)) *
              0.001);
}




int main(int argc, char** argv) {
  FILE* fi;
  FILE* outfile;
  int count, j, i, k;
  char poly_name[255];
  bool asc = true;

  cout << "\n=================================================================="
          "=========\n";
  cout << "\n                             INTERGEN PROJECT\n";
  cout << "\n                         (INTERferogramm GENerator)\n";
  cout << "\n                            Copyright BSU 2019\n";
  cout << "\n \n";
  cout << "\n To generate interferogramm based on subsidence and "
          "SAR-parameters\n";
  cout << "\n=================================================================="
          "=========\n";

  float a1, b1, c1, d1;

  // Input of configuration file
  ifstream infile("inferote.dat");
  // Prepare to call triangle.exe
  char** args = new char*[3];
  char buf1[50], buf2[50];
  int int_par[4];
  double double_par[7];
  string line;
  int index = 0;
  int double_index = 0;
  while (std::getline(infile, line)) {
    if (line.substr(0, 7) == "IntPar ") {
      int_par[index] = std::stoi(line.substr(7));
      index++;
    } else if (line.substr(0, 10) == "DoublePar ") {
      double_par[double_index] = std::stod(line.substr(10));
      double_index++;
    }
  }
  n_x_steps = int_par[0];   // разрешение по X
  n_y_steps = int_par[1];          // разрешение по Y
  first_day = int_par[2];   // день начала съемки
  second_day = int_par[3];         // день окочания съемки

  x_step = double_par[0];        // шаг модели по X
  y_step = double_par[1];               // шаг модели по Y
  lengthOfWave = double_par[2];  // длина волны
  xf_step = double_par[3];              // шаг съемки по X
  yf_step = double_par[4];       // шаг съемки по Y
  azimuth = double_par[5];                  // азимут
  inclination = double_par[6];           // наклон 

  XRE_SUMPL = x_step / xf_step;                           // самплинг X
  YRE_SUMPL = y_step / yf_step;                           // самплинг X
  n_xf_steps = (x_step * (n_x_steps - 1)) / xf_step + 1;  // размерность по X
  n_yf_steps = (y_step * (n_y_steps - 1)) / yf_step + 1;  // размерность по Y

  float tmp_min;

  xy_array1 = new float[n_x_steps * n_y_steps];
  xy_array2 = new float[n_xf_steps * n_yf_steps];

  // Сформируем оседания
  // Поверхность на первую дату
  sprintf_s(poly_name, "rmib/veliz/%d_iter.vtk", first_day);
  ifstream file(poly_name);
  std::string str;

   for (i = 0; i < n_x_steps; i++)
    for (j = 0; j < n_y_steps; j++) {
      std::getline(file, str);
      const char* ccpExample = str.c_str();
      sscanf_s(ccpExample, "%f", &c1);
      xy_array1[i * n_y_steps + j] = c1;
    }
  // Поверхность на вторую дату
  sprintf_s(poly_name, "./rmib/veliz/%d_iter.vtk", second_day);
  std::ifstream file2(poly_name);

  for (i = 0; i < n_x_steps; i++)
    for (j = 0; j < n_y_steps; j++) {
      std::getline(file2, str);
      const char* ccpExample = str.c_str();
      sscanf_s(ccpExample, "%f", &c1);
      xy_array1[i * n_y_steps + j] -= c1;
    }
  // палитра
  int pallete[17][3];
  pallete[0][0] = 219;
  pallete[0][1] = 223;
  pallete[0][2] = 82;
  pallete[1][0] = 239;
  pallete[1][1] = 235;
  pallete[1][2] = 51;
  pallete[2][0] = 239;
  pallete[2][1] = 197;
  pallete[2][2] = 88;
  pallete[3][0] = 239;
  pallete[3][1] = 161;
  pallete[3][2] = 124;
  pallete[4][0] = 239;
  pallete[4][1] = 124;
  pallete[4][2] = 161;
  pallete[5][0] = 239;
  pallete[5][1] = 88;
  pallete[5][2] = 197;
  pallete[6][0] = 239;
  pallete[6][1] = 51;
  pallete[6][2] = 235;
  pallete[7][0] = 197;
  pallete[7][1] = 88;
  pallete[7][2] = 239;
  pallete[8][0] = 161;
  pallete[8][1] = 124;
  pallete[8][2] = 239;
  pallete[9][0] = 124;
  pallete[9][1] = 161;
  pallete[9][2] = 239;
  pallete[10][0] = 88;
  pallete[10][1] = 197;
  pallete[10][2] = 239;
  pallete[11][0] = 51;
  pallete[11][1] = 235;
  pallete[11][2] = 239;
  pallete[12][0] = 88;
  pallete[12][1] = 239;
  pallete[12][2] = 235;
  pallete[13][0] = 124;
  pallete[13][1] = 239;
  pallete[13][2] = 197;
  pallete[14][0] = 161;
  pallete[14][1] = 239;
  pallete[14][2] = 161;
  pallete[15][0] = 197;
  pallete[15][1] = 239;
  pallete[15][2] = 124;
  pallete[16][0] = 235;
  pallete[16][1] = 239;
  pallete[16][2] = 88;

  // create image at first
  double res;
  double bi_val;
  int ttmp_ind;

  Mat rot_mat(2, 3, CV_32FC1);  // Матрица поворотов
  image = cvCreateImage(
      cvSize((n_x_steps - 1) * XRE_SUMPL, (n_y_steps - 1) * YRE_SUMPL),
      IPL_DEPTH_8U, 3);
  image_f = cvCreateImage(cvSize(n_xf_steps, n_yf_steps), IPL_DEPTH_8U, 3);
  CvScalar value;
  cvNamedWindow("original", CV_WINDOW_AUTOSIZE);
  for (j = 0; j < n_yf_steps - 1; j++)
    for (i = 0; i < n_xf_steps - 1; i++) {
      bi_val = xy_array2[i * n_yf_steps + j] =
          bi_val_1(i * xf_step, j * yf_step, asc);

      std::modf((bi_val * 2.0) / lengthOfWave, &res);
      ttmp_ind = 16 - 16 * fabs((bi_val * 2.0) / lengthOfWave - res);
      value.val[0] = 255 * fabs((bi_val * 2.0) / lengthOfWave - res);
      value.val[1] = 255 * fabs((bi_val * 2.0) / lengthOfWave - res);
      value.val[2] = 255 * fabs((bi_val * 2.0) / lengthOfWave - res);
      value.val[3] = 0.0;
      cvSet2D(image_f, j, i, value);
    }

  for (j = 0; j < n_yf_steps - 2; j++)
    for (i = 0; i < n_xf_steps - 2; i++) {
      bi_val = bi_int(i, j, asc);

      std::modf((bi_val * 2.0) / lengthOfWave,
                &res);
      ttmp_ind = 16 - 16 * fabs((bi_val * 2.0) / lengthOfWave -
                                res);
      value.val[0] = pallete[ttmp_ind][2];
      value.val[1] = pallete[ttmp_ind][1];
      value.val[2] = pallete[ttmp_ind][0];
      value.val[3] = 0.0;
      cvSet2D(image_f, j, i, value);
    }
  cvShowImage("original", image_f);
  cvWaitKey(0);	

  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
  compression_params.push_back(100);  // JPEG quality (0-100)

  cv::Mat matImage(image_f);
  cv::imwrite("result.jpg", matImage, compression_params);

  return 0;
}