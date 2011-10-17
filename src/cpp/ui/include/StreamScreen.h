#ifndef StreamScreen_H
#define StreamScreen_H

#include "qtall.h"

#include "DataStream.h"
#include "StreamTransformer.h"

class StreamScreen : public QGLWidget
{
//   Q_OBJECT        // must include this if you use Qt signals/slots

public:
   StreamScreen(QWidget *parent = 0);
   ~StreamScreen();

   void setTransformer(StreamTransformer* t);
   void refresh();

protected:
   StreamTransformer* transformer;

   void initializeGL();
   void resizeGL(int w, int h);
   void paintGL();
};

#endif

