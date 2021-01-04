#include "cnn.hpp"

using namespace std;
using namespace cv;


int main()
{
	Img img; 
	float* score;
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\bg.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\bg00.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\bg01.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\bg02.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\bg03.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\bg04.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face_child_male00.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face_man00.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face_man01.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face_man02.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face_woman00.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face_woman01.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	img.scanner("C:\\Users\\layer\\Desktop\\CNN\\face_woman02.jpg");
	score = img.facedect();
	printf("%.2f %.2f\n", score[0], score[1]);
	return 0;
}
