#include "cnn.hpp"

using namespace std;
using namespace cv;


int main()
{
	Img img;
	float* score;
	img.scanner("C:\\Users\\layer\\Desktop\\bg.jpg");
	score = img.facedect();
	cout << score[0] << endl;
	cout << score[1] << endl;
	return 0;
}