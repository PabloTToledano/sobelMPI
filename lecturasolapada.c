/* Pract2  RAP 09/10    Javier Ayllon*/

#include <openmpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h> 
#include <assert.h>   
#include <unistd.h> 
#include <stdio.h>
#include <math.h>

#define NIL (0)
#define PHOTO "foto.dat"
#define NPROCESS 4      
#define PHOTOCOLUMNS 400
#define PHOTOROWS 400
#define MAX_BRIGHTNESS 255
//Para cambiar el threshold del edge detection cambiar SIGMA, probado con 0.5,1,2,3
#define SIGMA 1

/*Variables Globales */

XColor colorX;
Colormap mapacolor;
char cadenaColor[]="#000000";
Display *dpy;
Window w;
GC gc;
/*Funciones auxiliares */

void initX() {

      dpy = XOpenDisplay(NIL);
      assert(dpy);

      int blackColor = BlackPixel(dpy, DefaultScreen(dpy));
      int whiteColor = WhitePixel(dpy, DefaultScreen(dpy));

      w = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0,
                                     PHOTOCOLUMNS, PHOTOROWS, 0, blackColor, blackColor);
      XSelectInput(dpy, w, StructureNotifyMask);
      XMapWindow(dpy, w);
      gc = XCreateGC(dpy, w, 0, NIL);
      XSetForeground(dpy, gc, whiteColor);
      for(;;) {
            XEvent e;
            XNextEvent(dpy, &e);
            if (e.type == MapNotify)
                  break;
      }


      mapacolor = DefaultColormap(dpy, 0);

}

void dibujaPunto(int x,int y, int r, int g, int b) {

        sprintf(cadenaColor,"#%.2X%.2X%.2X",r,g,b);
        XParseColor(dpy, mapacolor, cadenaColor, &colorX);
        XAllocColor(dpy, mapacolor, &colorX);
        XSetForeground(dpy, gc, colorX.pixel);
        XDrawPoint(dpy, w, gc,x,y);
        //XFlush(dpy);

}

void getPhotoPixels(MPI_Comm commPadre){
      int bufferPixelData[3];
      MPI_Status status;
      int photosize = PHOTOCOLUMNS*PHOTOROWS;
      for (size_t i = 0; i < photosize; i++)
      {
            MPI_Recv(&bufferPixelData,3,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,commPadre,&status);
            dibujaPunto(bufferPixelData[0],bufferPixelData[1],bufferPixelData[2],bufferPixelData[2],bufferPixelData[2]);

      }
      
      
}

int getStartRow(int rank){
      int rowsperworker = PHOTOROWS/NPROCESS;
      int startrow = rank * rowsperworker;

      return startrow;
}

int getEndRow(int rank, int startrow){
      int rowsperworker = PHOTOROWS/NPROCESS;
      int endrow = startrow + rowsperworker ;
      if(rank == NPROCESS-1)
            endrow = PHOTOROWS;
      
      return endrow;
}



int getReadRows(int rank){
      int readrows = PHOTOROWS/NPROCESS;
      if(rank == 0 ){
            readrows += 16;
      }else{
            readrows += 48;
      }
      return readrows;
}

static void convolution(const int *in,
            int       *out,
            const float   *kernel,
            const int      nx,
            const int      ny,
            const int      kn,
            const int     normalize){
      const int khalf = kn / 2;
      float min = 0.5;
      float max = 254.5;
      float pixel = 0.0;
      size_t c = 0;
      int m, n, i, j;

      assert(kn % 2 == 1);
      assert(nx > kn && ny > kn);

      for (m = khalf; m < nx - khalf; m++) {
      for (n = khalf; n < ny - khalf; n++) {
            pixel = c = 0;

            for (j = -khalf; j <= khalf; j++)
            for (i = -khalf; i <= khalf; i++)
            pixel += in[(n - j) * nx + m - i] * kernel[c++];

            if (normalize == 1)
            pixel = MAX_BRIGHTNESS * (pixel - min) / (max - min);

            out[n * nx + m] = (int) pixel;
      }
      }
}

void gaussian_filter(const int *in,
                int       *out,
                const int      nx,
                const int      ny,
                const float    sigma){
      const int n = 2 * (int) (2 * sigma) + 3;
      const float mean = (float) floor(n / 2.0);
      float kernel[n * n];
      int i, j;
      size_t c = 0;

      for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
            kernel[c++] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);
      }

      convolution(in, out, kernel, nx, ny, n, 1);
}

void edgeDetection(int rank,MPI_Comm commPadre){
      MPI_File photo;
      MPI_Status status;
      int bufferPixelData[3];
      int red,green,blue;

      int startrow = getStartRow(rank);
      int endrow = getEndRow(rank,startrow);
      int readrows = getReadRows(rank);

      
      unsigned char *photoData = malloc(3*readrows*PHOTOCOLUMNS*sizeof(unsigned char)); //falta multiplicar los pixeles que voy a necesitar
      int *grayScaleIMG = calloc(readrows*PHOTOCOLUMNS, sizeof(int));      
      int *blurIMG = calloc(readrows*PHOTOCOLUMNS, sizeof(int));
      int *edgeImgY = calloc(readrows*PHOTOCOLUMNS, sizeof(int));
      int *edgeImgX = calloc(readrows*PHOTOCOLUMNS, sizeof(int));
      MPI_Offset point =  ( ( ( rank* (PHOTOROWS/NPROCESS)-32 )) *sizeof(unsigned char)*3*PHOTOCOLUMNS );
      if(rank == 0){
           point= 0 ;
      }

      MPI_File_open(MPI_COMM_WORLD,PHOTO,MPI_MODE_RDONLY,MPI_INFO_NULL,&photo);
      MPI_File_set_view(photo, point, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL); 
      MPI_File_read(photo,photoData,3*readrows*PHOTOCOLUMNS,MPI_UNSIGNED_CHAR,&status);
      printf("Soy: %d voy a leer, mi offset: %lld\n",rank,point);
      for (int i = 0; i < readrows; i++)
      {
            for (int j = 0; j < PHOTOCOLUMNS; j++)
            {
                  //pixel RGB data
                  red = (int)*photoData;
                  photoData = photoData + sizeof(unsigned char);
                  green = (int)*photoData;
                  photoData = photoData + sizeof(unsigned char);
                  blue = (int)*photoData;
                  photoData = photoData + sizeof(unsigned char);
                  grayScaleIMG[i*PHOTOCOLUMNS+j] = red*.2+ green*.7+ blue*.1;                  
                  
            }
      }
      MPI_File_close(&photo);
      printf("Tengo imagen %d\n",rank);
      const float Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
      const float Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

      gaussian_filter(grayScaleIMG,blurIMG,PHOTOCOLUMNS,readrows,SIGMA);
      convolution(blurIMG, edgeImgX, Gx, PHOTOCOLUMNS, readrows, 3, 0);
      convolution(blurIMG, edgeImgY, Gy, PHOTOCOLUMNS, readrows, 3, 0);
      printf("Aplique el filtro %d\n",rank);
      int x = 0;
      int y = startrow;
      int startread = 0;
      if(rank != 0 ){
            startread = 32;
      }


      for (int i = startread; i < ((PHOTOROWS/NPROCESS)+startread); i++)
      {
            for (int j = 0; j < PHOTOCOLUMNS; j++)
            {
                  bufferPixelData[0] = x;
                  bufferPixelData[1] = y;
                  float temp = 0;
                  temp = sqrt(edgeImgX[i*PHOTOCOLUMNS+j]*edgeImgX[i*PHOTOCOLUMNS+j] + edgeImgY[i*PHOTOCOLUMNS+j]*edgeImgY[i*PHOTOCOLUMNS+j]);
                  bufferPixelData[2] = temp;
                  
                  //printf("Envio el: %d %d\n",y,x);
                  MPI_Send(&bufferPixelData,3,MPI_INT,0,1,commPadre);
                  x = x+1;
            }
            x=0;
            y = y+1;
      }

      
     

}

/* Programa principal */

int main (int argc, char *argv[]) {

  int rank,size;
  MPI_Comm commPadre;
  int tag;
  MPI_Status status;
  int buf[5];
  int errCodes[NPROCESS];


  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_get_parent( &commPadre );
  if ( (commPadre==MPI_COMM_NULL)
        && (rank==0) )  {

	initX();
      sleep(1);
	/* Codigo del maestro */
      MPI_Comm_spawn("./pract2",MPI_ARGV_NULL,NPROCESS,MPI_INFO_NULL,0,MPI_COMM_WORLD,&commPadre,errCodes);
	/*En algun momento dibujamos puntos en la ventana algo como
	  */
      getPhotoPixels(commPadre);
      sleep(10000);

  }

 	
  else {
    /* Codigo de todos los trabajadores */
    /* El archivo sobre el que debemos trabajar es foto.dat */
      edgeDetection(rank,commPadre);

  }

  MPI_Finalize();

}

