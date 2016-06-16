
#include <cstdlib>
#include <iostream>
#include <random>
#include <fstream>

using namespace std;

int main(int argc, char **argv){
	if (argc < 6)
		return (cout << "Usage: randmat rows cols max min filename" << endl),1;

	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);
	int maxval = atoi(argv[3]);
	int minval = atoi(argv[4]);

	if (minval > maxval)
		swap(minval,maxval);

	ofstream file;
	file.open(argv[5]);

	if (!file.is_open())
		return (cerr << "Error opening file: " << argv[5] << endl),1;

	file << rows << " " << cols << endl;

	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<float> rndval(minval,maxval);

	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			file << rndval(mt) << " ";
		}
		file << endl;
	}

	file.close();

	return 0;
}
