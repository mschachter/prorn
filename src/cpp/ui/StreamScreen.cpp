#include <stdio.h>

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "StreamScreen.h"   

StreamScreen::StreamScreen(QWidget *parent) : QGLWidget(parent)
{
   printf("StreamScreen created.\n");
   transformer = NULL;
}

StreamScreen::~StreamScreen()
{
   printf("StreamScreen destroyed.\n");
}

void StreamScreen::initializeGL()
{
   glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
}

void StreamScreen::resizeGL(int w, int h)
{
   printf("resizeGL called, w=%d, h=%d\n", w, h);
}

void StreamScreen::paintGL()
{
   glClear(GL_COLOR_BUFFER_BIT);

   if (transformer != NULL) {
      glDrawPixels(width(), height(), GL_LUMINANCE, GL_UNSIGNED_BYTE, transformer->transform());
   } else {
      printf("Transformer is null!\n");
   }

   glFlush();
}


void StreamScreen::setTransformer(StreamTransformer* t)
{
   transformer = t;
}


void StreamScreen::refresh()
{
   printf("Refreshing stream screen...\n");
}
