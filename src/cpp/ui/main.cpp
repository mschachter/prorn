#include "StreamScreen.h"
#include "DataStream.h"

const int WIDTH = 640;
const int HEIGHT = 480;

QPushButton* create_quit_button(QApplication* app, QWidget* window)
{
   QPushButton* quit = new QPushButton("Quit", window);
   quit->setFont(QFont("Times", 12, QFont::Bold)); 
   
   int bwidth = 100;
   int bheight = 30;
   int bx = 320 - (bwidth / 2);
   int by = 570 - bheight;

   quit->setGeometry(bx, by, bwidth, bheight);
   QObject::connect(quit, SIGNAL(clicked()), app, SLOT(quit()));

   return quit;
}

int main(int argc, char *argv[])
{
   RandomStreamInt randStream(HEIGHT, WIDTH, 150, 20);
   randStream.poll(true);
   StreamTransformer2d trans(&randStream);
   QApplication app(argc, argv);

   QWidget window;
   window.resize(640, 580);

   QPushButton* quit = create_quit_button(&app, &window);

   StreamScreen* glWin = new StreamScreen(&window);
   glWin->setGeometry(0, 0, WIDTH, HEIGHT);
   glWin->setTransformer(&trans);

   window.show();
   return app.exec();
}


