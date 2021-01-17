/* Pract2  RAP 09/10    Javier Ayllon*/

#include <openmpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h> 
#include <assert.h>   
#include <unistd.h> 
#include <stdio.h>
#include <math.h>
#include <string.h>

#define NIL (0)
#define PHOTO "foto.dat"
#define NPROCESS 4      
#define PHOTOCOLUMNS 400
#define PHOTOROWS 400
#define MAX_BRIGHTNESS 255
//Para cambiar el threshold del edge detection cambiar SIGMA, probado con 0.5,1,2
#define SIGMA 0.5
#define FILTERROWS 10

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
      printf("Gato getto daze\n");
      
}

int getStartRow(int rank){
      int rowsperworker = PHOTOROWS/NPROCESS;
      int startrow = rank * rowsperworker;

      return startrow;
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

int getReadRows(int rank){
      int readrows = PHOTOROWS/NPROCESS;
      if(rank == 0 || rank== NPROCESS-1){
            readrows += FILTERROWS;
      }else{
            readrows += FILTERROWS*2;
      }
      return readrows;
}

void printarray(int *array){
      //funcion super cutre girald no mires
      printf("%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n\n",array[0],array[1],array[2],array[3],array[4],array[5],array[6],
      array[7],array[8],array[9],array[10],array[11],array[12],array[13],array[14],array[15],array[16],array[17],array[18],array[19],array[20],array[21],array[22],
      array[23],array[24]);
}

void topBorderPixels(int rank, MPI_File photo,MPI_Comm commPadre){
      int rowsperworker  = PHOTOROWS/NPROCESS;
      int red,green,blue;
      MPI_Status statusInf,statusSup,status;
      MPI_Request requestInf, requestSup;
      MPI_Comm comm = MPI_COMM_WORLD;
      int sendPixel,recPixel;
      unsigned char photoData[3];
      int bufferPixelData[3];
      int *edgeImgY = calloc(24, sizeof(int));
      int *edgeImgX = calloc(24, sizeof(int));
      int *grayScale = calloc(24, sizeof(int));
      const float Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
      const float Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
      //primera linea
      int x=0;
      int y=getStartRow(rank);
      MPI_Offset offsetEnvio = PHOTOCOLUMNS*3*(rowsperworker-2);;
      MPI_Offset offsetRead =  0;
      for(int i=0;i<PHOTOCOLUMNS-4;i++){
            if(rank != 0){
                  for(int k = 2;k<5;k++){
                        
                        for(int l = 0;l<5;l++){
                              MPI_File_read_at(photo,offsetRead,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                              red = (int)photoData[0];
                              green = (int)photoData[1];
                              blue = (int)photoData[2];
                              grayScale[k*5+l] = red*.2+ green*.7+ blue*.1;                             
                              offsetRead = offsetRead + 3;
                        }
                        offsetRead = offsetRead + 3*PHOTOCOLUMNS - 3*5; //avanzar a la linea siguiente al principio
                  }
                  offsetRead = offsetRead +  -3*3*PHOTOCOLUMNS + 3;
            }
            for(int k = 0;k<2;k++){ 
                  for(int l = 0;l<5;l++){
                        if(rank !=NPROCESS-1){
                              MPI_File_read_at(photo,offsetEnvio,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                              red = (int)photoData[0];
                              green = (int)photoData[1];
                              blue = (int)photoData[2];
                              sendPixel = red*.2+ green*.7+ blue*.1; 
                              offsetEnvio = offsetEnvio + 3;
                              MPI_Isend(&sendPixel,1,MPI_INT,rank+1,0,comm,&requestSup);
                        }
                        if(rank !=0){
                              MPI_Irecv(&recPixel,1,MPI_INT,rank-1,0,comm,&requestSup);
                              MPI_Wait(&requestSup,&statusSup);
                              grayScale[k*5+l] = recPixel;
                        }
                        
                  }
                  if(rank !=NPROCESS-1)
                  offsetEnvio = offsetEnvio + 3*PHOTOCOLUMNS - 3*5;
            }
            if(rank !=NPROCESS-1)
            offsetEnvio = offsetEnvio +  -2*3*PHOTOCOLUMNS + 3;
            if(rank !=0){
            convolution(grayScale, edgeImgX, Gx, 5, 5, 3, 0);
            convolution(grayScale, edgeImgY, Gy, 5, 5, 3, 0);
            float temp = 0;
            temp = sqrt(edgeImgX[12]*edgeImgX[12] + edgeImgY[12]*edgeImgY[12]);
            bufferPixelData[0] = x;
            bufferPixelData[1] = y;
            bufferPixelData[2] = temp;
            x = x + 1;
            MPI_Send(&bufferPixelData,3,MPI_INT,0,1,commPadre);
            //printarray(grayScale);
            }
      }  
       //segunda linea
      offsetRead = 0;
      offsetEnvio = PHOTOCOLUMNS*3*(rowsperworker-1);
      x= 0;
      y = getStartRow(rank)+1;
      for(int i=0;i<PHOTOCOLUMNS-4;i++){
            if(rank != 0){
                  for(int k = 1;k<5;k++){
                        
                        for(int l = 0;l<5;l++){
                              MPI_File_read_at(photo,offsetRead,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                              red = (int)photoData[0];
                              green = (int)photoData[1];
                              blue = (int)photoData[2];
                              grayScale[k*5+l] = red*.2+ green*.7+ blue*.1;                             
                              offsetRead = offsetRead + 3;
                        }
                       
                        offsetRead = offsetRead + 3*PHOTOCOLUMNS - 3*5; //avanzar a la linea siguiente al principio
                  }
                  offsetRead = offsetRead +  -4*3*PHOTOCOLUMNS + 3;
            }
             
            for(int l = 0;l<5;l++){
                  if(rank !=NPROCESS-1){
                        MPI_File_read_at(photo,offsetEnvio,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                        red = (int)photoData[0];
                        green = (int)photoData[1];
                        blue = (int)photoData[2];
                        sendPixel = red*.2+ green*.7+ blue*.1; 
                        offsetEnvio = offsetEnvio + 3;
                        MPI_Isend(&sendPixel,1,MPI_INT,rank+1,0,comm,&requestInf); 
                  }
                  if(rank != 0){
                        MPI_Irecv(&recPixel,1,MPI_INT,rank-1,0,comm,&requestInf);
                        MPI_Wait(&requestInf,&statusInf);
                        grayScale[l] = recPixel;
                  }
            }
            if(rank != 0){
            convolution(grayScale, edgeImgX, Gx, 5, 5, 3, 0);
            convolution(grayScale, edgeImgY, Gy, 5, 5, 3, 0);
            float temp = 0;
            temp = sqrt(edgeImgX[12]*edgeImgX[12] + edgeImgY[12]*edgeImgY[12]);
            bufferPixelData[0] = x;
            bufferPixelData[1] = y;
            bufferPixelData[2] = temp;
            x = x + 1;
            MPI_Send(&bufferPixelData,3,MPI_INT,0,1,commPadre);
            //printarray(grayScale);

            }
      }
  
}

void bottomBorderPixels(int rank, MPI_File photo,MPI_Comm commPadre){
      int rowsperworker  = PHOTOROWS/NPROCESS;
      int red,green,blue;
      MPI_Status status;
      MPI_Status statusInf,statusSup;
      MPI_Request requestInf, requestSup;
      MPI_Comm comm = MPI_COMM_WORLD;
      int sendPixel;
      int recPixel;
      unsigned char photoData[3];
      int bufferPixelData[3];
      int *edgeImgY = calloc(25, sizeof(int));
      int *edgeImgX = calloc(25, sizeof(int));
      int *grayScale = calloc(24, sizeof(int));
      const float Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
      const float Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
      //penultima fila
      MPI_Offset offsetRead = PHOTOCOLUMNS*3*(rowsperworker-4); //-4
      MPI_Offset offsetEnvio = 0;
      int x = 0;
      int y = rowsperworker + getStartRow(rank) - 2;
      for(int i=0;i<PHOTOCOLUMNS-4;i++){
            if(rank!=NPROCESS-1){
                  for(int k = 0;k<4;k++){
                        
                        for(int l = 0;l<5;l++){
                              MPI_File_read_at(photo,offsetRead,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                              red = (int)photoData[0];
                              green = (int)photoData[1];
                              blue = (int)photoData[2];
                              grayScale[k*5+l] = red*.2+ green*.7+ blue*.1;                             
                              offsetRead = offsetRead + 3;                              
                        }                       
                        offsetRead = offsetRead + 3*PHOTOCOLUMNS - 3*5; //avanzar a la linea siguiente al principio
                  }
                  offsetRead = offsetRead +  -4*3*PHOTOCOLUMNS + 3;
            }
            for(int l = 0;l<5;l++){
                  if(rank != 0){
                        MPI_File_read_at(photo,offsetEnvio,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                        red = (int)photoData[0];
                        green = (int)photoData[1];
                        blue = (int)photoData[2];
                        sendPixel = red*.2+ green*.7+ blue*.1; 
                        offsetEnvio = offsetEnvio + 3;
                        MPI_Isend(&sendPixel,1,MPI_INT,rank-1,1,comm,&requestSup);
                  }
                  if(rank !=NPROCESS-1){
                        MPI_Irecv(&recPixel,1,MPI_INT,rank+1,1,comm,&requestSup);
                        MPI_Wait(&requestSup,&statusSup);
                        grayScale[20+l] = recPixel;
                  }
            }
            if(rank!=NPROCESS-1){
            convolution(grayScale, edgeImgX, Gx, 5, 5, 3, 0);
            convolution(grayScale, edgeImgY, Gy, 5, 5, 3, 0);
            float temp = 0;
            temp = sqrt(edgeImgX[12]*edgeImgX[12] + edgeImgY[12]*edgeImgY[12]);
            bufferPixelData[0] = x;
            bufferPixelData[1] = y;
            bufferPixelData[2] = temp;
            x = x + 1;
            MPI_Send(&bufferPixelData,3,MPI_INT,0,1,commPadre);
            //printarray(grayScale);

            }
      }
      //Ultima Linea
      x=0;
      y=rowsperworker + getStartRow(rank) - 1;
      offsetEnvio = 0;
      offsetRead = PHOTOCOLUMNS*3*(rowsperworker-2); //penultima fila
      for(int i=0;i<PHOTOCOLUMNS-4;i++){
            if(rank!=NPROCESS-1){
                  for(int k = 0;k<3;k++){
                        
                        for(int l = 0;l<5;l++){
                              MPI_File_read_at(photo,offsetRead,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                              red = (int)photoData[0];
                              green = (int)photoData[1];
                              blue = (int)photoData[2];
                              grayScale[k*5+l] = red*.2+ green*.7+ blue*.1;                             
                              offsetRead = offsetRead + 3;
                        }
                        offsetRead = offsetRead + 3*PHOTOCOLUMNS - 3*5; //avanzar a la linea siguiente al principio
                  }
                  offsetRead = offsetRead +  -3*3*PHOTOCOLUMNS + 3;
            }
            for(int k = 3;k<5;k++){ 
                  for(int l = 0;l<5;l++){
                        if(rank != 0){
                              MPI_File_read_at(photo,offsetEnvio,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                              red = (int)photoData[0];
                              green = (int)photoData[1];
                              blue = (int)photoData[2];
                              sendPixel = red*.2+ green*.7+ blue*.1; 
                              offsetEnvio = offsetEnvio + 3;
                              MPI_Isend(&sendPixel,1,MPI_INT,rank-1,l+k,comm,&requestSup);
                        }
                        if(rank !=NPROCESS-1){
                              MPI_Irecv(&recPixel,1,MPI_INT,rank+1,l+k,comm,&requestSup);
                              MPI_Wait(&requestSup,&statusSup);
                              grayScale[k*5+l] = recPixel;
                        }    
                  }
                  if(rank != 0)
                  offsetEnvio = offsetEnvio + 3*PHOTOCOLUMNS - 3*5;
            }
            if(rank != 0)
            offsetEnvio = offsetEnvio +  -2*3*PHOTOCOLUMNS + 3;
            if(rank!=NPROCESS-1){
            convolution(grayScale, edgeImgX, Gx, 5, 5, 3, 0);
            convolution(grayScale, edgeImgY, Gy, 5, 5, 3, 0);
            float temp = 0;
            temp = sqrt(edgeImgX[12]*edgeImgX[12] + edgeImgY[12]*edgeImgY[12]);
            bufferPixelData[0] = x;
            bufferPixelData[1] = y;
            bufferPixelData[2] = temp;
            x = x + 1;
            MPI_Send(&bufferPixelData,3,MPI_INT,0,1,commPadre);
            //printarray(grayScale);

            }
      
      }
     
} 

void centerPixels(int rank,MPI_File photo,MPI_Comm commPadre){
      MPI_Status status;
      int x = 0;
      int y = getStartRow(rank)+2;
      int readrows = getReadRows(rank);
      int rowsperworker  = PHOTOROWS/NPROCESS;
      int red,green,blue;
      MPI_Offset offsetRead = 0;
      unsigned char photoData[3];
      int *grayScale = calloc(24, sizeof(int));
      const float Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
      const float Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
      int *edgeImgY = calloc(25, sizeof(int));
      int *edgeImgX = calloc(25, sizeof(int));
      int bufferPixelData[3];
      for (int i = 0; i < rowsperworker-4; i++) //ignoro tanto las 2 primeras como 2 ultimas
      {
           offsetRead = i*3*PHOTOCOLUMNS ;
            for (int j = 0; j < PHOTOCOLUMNS-4; j++)
            {

                  for(int k = 0;k<5;k++){
                        
                        for(int l = 0;l<5;l++){
                              MPI_File_read_at(photo,offsetRead,&photoData,3,MPI_UNSIGNED_CHAR,&status);
                              //MPI_File_read(photo, &photoData,3, MPI_UNSIGNED_CHAR, &status);
                              red = (int)photoData[0];
                              green = (int)photoData[1];
                              blue = (int)photoData[2];
                              grayScale[k*5+l] = red*.2+ green*.7+ blue*.1;                             
                              offsetRead = offsetRead + 3;
                              //MPI_File_seek(photo,  offsetRead, MPI_SEEK_CUR);

                        }
                       
                        offsetRead = offsetRead + 3*PHOTOCOLUMNS - 3*5; //avanzar a la linea siguiente al principio
                  }

                  convolution(grayScale, edgeImgX, Gx, 5, 5, 3, 0);
                  convolution(grayScale, edgeImgY, Gy, 5, 5, 3, 0);
                  float temp = 0;
                  temp = sqrt(edgeImgX[12]*edgeImgX[12] + edgeImgY[12]*edgeImgY[12]);
                  bufferPixelData[0] = x;
                  bufferPixelData[1] = y;
                  bufferPixelData[2] = temp;
                  //printf("Envio el: %d %d\n",y,x);
                  MPI_Send(&bufferPixelData,3,MPI_INT,0,1,commPadre);
                  //pixel RGB data
                  x = x+1;
                  offsetRead = offsetRead +  -5*3*PHOTOCOLUMNS + 3;
                  

            }
            x=0;
            y = y+1;
      }
}

void edgeDetection(int rank,MPI_Comm commPadre){

      int rowsperworker  = PHOTOROWS/NPROCESS;
      MPI_File photo;
      MPI_Offset point = rank*rowsperworker*sizeof(unsigned char)*3*PHOTOROWS;

      MPI_File_open(MPI_COMM_WORLD,PHOTO,MPI_MODE_RDONLY,MPI_INFO_NULL,&photo);
      MPI_File_set_view(photo, point, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL);

      topBorderPixels(rank,photo,commPadre);

      centerPixels(rank,photo,commPadre);

      bottomBorderPixels(rank,photo,commPadre);

      MPI_File_close(&photo);

      
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

