
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <opencv/cv.h>

using namespace cv;

const int WIDTH = 640;
const int HEIGHT = 480;

Mat img;

void RenderScene(void)
{
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, img.data);
   glFlush();
}


void SetupRC(void)
{
   glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
}

void stripe(Mat &img)
{
  for (int k = 0; k < img.rows; k++) {
      uchar* row = img.ptr<uchar>(k);
      if (k % 10 == 0) {
         for (int j = 0; j < img.cols; j++) {
            row[j] = 255;
            //img.at<uchar>(j, k));
         }
      }
   }
}


int main(int argc, char **argv)
{
   int nrows = HEIGHT;
   int ncols = WIDTH;
   img = Mat(Size(ncols, nrows), CV_8UC1);
   printf("Image size: %dx%d, (%d rows, %d cols)\n", img.size[0], img.size[1], img.rows, img.cols);
   randn(img, Scalar::all(125), Scalar::all(10));
   
   stripe(img);

   /*
   for (int k = 0; k < nrows; k++) {
      for (int j = 0; j < ncols; j++) {
         int l = k*ncols + j;
         printf("img[%d][%d]=%d, img[%d]=%d\n", k, j, img.at<uchar>(j, k), l, img.data[l]);
      }
   }
   */

   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
   glutInitWindowSize(WIDTH, HEIGHT);
   glutInitWindowPosition(0, 0);
   glutCreateWindow("GL Test");
   glutDisplayFunc(RenderScene);

   SetupRC();

   glutMainLoop();

   return 0;
}

