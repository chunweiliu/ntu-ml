#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>	/* for padding output */
#define TEST_SIZE 1000 	/* for testing set */
#define TRAIN_SIZE 100	/* for training set */
#define ITERATION 1000
#define VEC_SIZE 3	/* size of x(n+1) */
using namespace std;

class vec3 {
	public:
		double x[VEC_SIZE];
		inline vec3 () {
			x[0] = -1.0;	/* expend each x(n) to x(n+1) */
			for ( int i = 1; i < VEC_SIZE; i++ ) {
				x[i] = 0.0;
			}
		}
		~vec3 () {}
		inline vec3( double x_, double y_, double z_ ){
			x[0] = x_;
			x[1] = y_;
			x[2] = z_;
		}

		inline vec3& operator = (const vec3 &rhs) {
			x[0] = rhs.x[0];
			x[1] = rhs.x[1];
			x[2] = rhs.x[2];
		}

		inline const double operator [] (int index) const {
			return ((double*)(&x))[index];
		}

		inline const vec3 operator + (const vec3 &rhs) const {
			return vec3(x[0] + rhs[0], x[1] + rhs[1], x[2] + rhs[2]); 
		}

		inline const vec3 operator - (const vec3 &rhs) const {
			return vec3(x[0] - rhs[0], x[1] - rhs[1], x[2] - rhs[2]); 
		}

		inline const vec3 operator * (const double d) const {
			return vec3(x[0] * d, x[1] * d, x[2] * d);
		}

		inline double dot(const vec3 &rhs) const {
			return x[0]*rhs[0] + x[1]*rhs[1] + x[2]*rhs[2];
		}

		inline vec3& operator += (const vec3 &rhs) {
			x[0] += rhs[0];
			x[1] += rhs[1];
			x[2] += rhs[2];
			return *this;
		}

		inline double norm () const {
			return sqrt( x[0]*x[0] + x[1]*x[1] + x[2]*x[2] );
		}

		inline void  normalise () {
			const double invlen = 1.0 / norm();
			x[0] *= invlen;
			x[1] *= invlen;
			x[2] *= invlen;
		}
};

void readfile ( ifstream &input, vec3 *x, double *y ) {

	int index = 1;
	string s;
	while ( !input.eof() ) {

		for ( int i = 1; i < VEC_SIZE; ++i) {
			input >> s;
			x[ index ].x[i] = atof( s.c_str() );
		}
		input  >> s; 	/* the last input of one line is y */
		y[ index ] = atof( s.c_str() );

		index++;
	}
	input.close();
}

void plr_noisy( bool flag, vec3 *train_x, double *train_y, vec3 *test_x, double *test_y, 
		vec3 *g, double *v, double *pi ) 
{
	/* 
	 * IF FLAG = 0, IT JSUT A NORMAL PLR 
	 * IF FLAG = 1, IT WILL USE THE BEST G FOR G* 
	 */

	/* PERCEPTRON LEARNING RULE */
	srand( (unsigned)time(0) );
	vec3 w(0.0, 0.0, 0.0);				// w is the coeff. of the decision function

	double best_v = 1.0;
	vec3 best_w(0.0, 0.0, 0.0);

	/* TRAINING BUT JUST PICK THE BEST TRAINING RESULT FOR G* */
	for ( int i = 1; i <= ITERATION; i++ ) {

		/* chose a random number from training set */
		int r = ( rand() % TRAIN_SIZE ) + 1; // from 1 to 100 

		double d = w.dot(train_x[r]);

		/* compare with the sign between y and <w, x> */
		bool sign_equal = ((( train_y[r] > 0 ) - ( train_y[r] < 0 )) == ((d > 0) - (d < 0)));
		if ( !sign_equal ) { 
			w += (train_x[r] * train_y[r]); // trainang mode, w have to been tuning 
		}

		if ( flag == 1 ) {
			/* TRAINING ERROR of g */
			double tmp_v = 0.0;
			for ( int j = 1; j <= TRAIN_SIZE ; j++ ) {

				double d = w.dot(train_x[j]);

				bool sign_equal = ((( train_y[j] > 0 ) - ( train_y[j] < 0 )) == ((d > 0) - (d < 0)));

				if ( !sign_equal ) {
					tmp_v += 1.0;
				}

			}
			tmp_v /= TRAIN_SIZE;

			/* chose the best g for g*(i) */
			if ( tmp_v < best_v ) {
				best_w = w;
				best_v = tmp_v;
			}
			g[i] = best_w;
		}
		else {
			g[i] = w;
		}

		//* for a g(i), we got to use it to find out the error */
		/* TRAINING ERROR */
		v[i] = 0.0;
		for( int j = 1; j <= TRAIN_SIZE; j++ ) {

			double d = g[i].dot(train_x[j]);

			bool sign_equal = ((( train_y[j] > 0 ) - ( train_y[j] < 0 )) == ((d > 0) - (d < 0)));

			if ( !sign_equal ) {
				v[i] += 1.0;
			}
		}
		v[i] /= TRAIN_SIZE; /* training error */

		/* TESTING ERROR */
		pi[i] = 0.0;
		for( int j = 1; j <= TEST_SIZE; j++ ) {

			double d = g[i].dot(test_x[j]);

			bool sign_equal = ((( test_y[j] > 0 ) - ( test_y[j] < 0 )) == ((d > 0) - (d < 0)));

			if ( !sign_equal ) {
				pi[i] += 1.0;
			}
		}
		pi[i] /= TEST_SIZE;
	}
}

int main ( int argc, char *argv[] )
{
	if (argc == 1) { 

		/* open the training file */
		ifstream training;
		training.open("data/hw1_3_train.dat", ios::in); // "filename" -> filename

		/* open the testing file */
		ifstream testing;
		testing.open("data/hw1_3_test.dat", ios::in);

		/* open the noisy training file */
		ifstream training_noisy;
		training_noisy.open("data/hw1_4_train.dat", ios::in); // "filename" -> filename

		/* open the noisy testing file */
		ifstream testing_noisy;
		testing_noisy.open("data/hw1_4_test.dat", ios::in);

		if ( training.is_open() && testing.is_open() && training_noisy.is_open() && testing_noisy.is_open() ) {

			/* loading the input stream from the file*/
			vec3 *train_x = new vec3[ TRAIN_SIZE + 2]; // 0 1 2 3 ... 1000 1001
			double *train_y = new double[ TRAIN_SIZE + 2];
			readfile ( training, train_x, train_y );

			vec3 *test_x = new vec3[ TEST_SIZE + 2];
			double *test_y = new double[ TEST_SIZE + 2];
			readfile ( testing, test_x, test_y );

			vec3 *train_x_noisy = new vec3[ TRAIN_SIZE + 2]; // 0 1 2 3 ... 1000 1001
			double *train_y_noisy = new double[ TRAIN_SIZE + 2];
			readfile ( training_noisy, train_x_noisy, train_y_noisy );

			vec3 *test_x_noisy = new vec3[ TEST_SIZE + 2];
			double *test_y_noisy = new double[ TEST_SIZE + 2];
			readfile ( testing_noisy, test_x_noisy, test_y_noisy );

			/* run the plr training algorithm iteratively */
			/* 1.3(1) 1.3(2) */
			int flag = 0;
			vec3 *g = new vec3[ ITERATION + 2];		/* g use to store the w(t) */
			double *v = new double[ ITERATION + 2]; 	/* v is the error rate of g of the training data */
			double *pi = new double[ ITERATION + 2]; 	/* pi is the error rate of g of the testing data */	
			plr_noisy ( flag, train_x, train_y, test_x, test_y, g, v, pi );

			/* run the plr* algorithm */
			/* 1.4(1) */
			flag = 0;
			vec3 *g_star_0 = new vec3[ ITERATION + 2];
			double *v_star_0 = new double[ ITERATION + 2]; 	/* vstar is the error rate of gstar of the training data */
			double *pi_star_0 = new double[ ITERATION + 2]; /* pistar is the error rate of gstar of the testing data */	
			plr_noisy ( flag, train_x_noisy, train_y_noisy, test_x_noisy, test_y_noisy, g_star_0, v_star_0, pi_star_0 );
			
			/* run the plr* algorithm */
			/* 1.4(2) */
			flag = 1;
			vec3 *g_star_1 = new vec3[ ITERATION + 2];
			double *v_star_1 = new double[ ITERATION + 2]; 	/* vstar is the error rate of gstar of the training data */
			double *pi_star_1 = new double[ ITERATION + 2];	/* pistar is the error rate of gstar of the testing data */	
			plr_noisy ( flag, train_x_noisy, train_y_noisy, test_x_noisy, test_y_noisy, g_star_1, v_star_1, pi_star_1 );


			/* cout the train function g(1000) and g_star(1000) to verify the result correct or not */
			vec3 nw;
			nw = g[ ITERATION ];
			nw.normalise();

			vec3 nw_star_0;
			nw_star_0 = g_star_0[ ITERATION ];
			nw_star_0.normalise();

			vec3 nw_star_1;
			nw_star_1 = g_star_1[ ITERATION ];
			nw_star_1.normalise();
			
			cout << "g    = ( " << setw(9) << nw.x[0] << " " << nw.x[1] << " " << nw.x[2] << " )" << endl; 
			cout << "g*_0 = ( " << setw(9) << nw_star_0.x[0] << " " << nw_star_0.x[1] << " " << nw_star_0.x[2] << " )" << endl; 
			cout << "g*_1 = ( " << setw(9) << nw_star_1.x[0] << " " << nw_star_1.x[1] << " " << nw_star_1.x[2] << " )" << endl; 

			/* output our result */
			ofstream train_out;
			train_out.open("train.txt", ios::out);

			ofstream test_out;
			test_out.open("test.txt", ios::out);

			ofstream w_out;
			w_out.open("w.txt", ios::out);

			ofstream train_out_noisy_0;
			train_out_noisy_0.open("train_n_0.txt", ios::out);

			ofstream test_out_noisy_0;
			test_out_noisy_0.open("test_n_0.txt", ios::out);

			ofstream w_out_noisy_0;
			w_out_noisy_0.open("w_n_0.txt", ios::out);

			ofstream train_out_noisy_1;
			train_out_noisy_1.open("train_n_1.txt", ios::out);

			ofstream test_out_noisy_1;
			test_out_noisy_1.open("test_n_1.txt", ios::out);

			ofstream w_out_noisy_1;
			w_out_noisy_1.open("w_n_1.txt", ios::out);

			for ( int i = 1; i <= ITERATION; i++ ) {
				train_out << i << " " << v[i] << endl; 
				test_out << i << " " << pi[i] << endl;
				w_out << i << " " << g[i].x[0] << " " << g[i].x[1] << " " << g[i].x[2] << endl;

				train_out_noisy_0 << i << " " << v_star_0[i] << endl; 
				test_out_noisy_0 << i << " " << pi_star_0[i] << endl;
				w_out_noisy_0 << i << " " << g_star_0[i].x[0] << " " << g_star_0[i].x[1] << " " << g_star_0[i].x[2] << endl;
				
				train_out_noisy_1 << i << " " << v_star_1[i] << endl; 
				test_out_noisy_1 << i << " " << pi_star_1[i] << endl;
				w_out_noisy_1 << i << " " << g_star_1[i].x[0] << " " << g_star_1[i].x[1] << " " << g_star_1[i].x[2] << endl;
			}

			train_out.close();
			test_out.close();
			w_out.close();
			train_out_noisy_0.close();
			test_out_noisy_0.close();
			w_out_noisy_0.close();
			train_out_noisy_1.close();
			test_out_noisy_1.close();
			w_out_noisy_1.close();

			delete []train_x;
			delete []train_y;
			delete []test_x;
			delete []test_y;
			delete []g;
			delete []v;
			delete []pi;
			delete []g_star_0;
			delete []v_star_0;
			delete []pi_star_0;
			delete []g_star_1;
			delete []v_star_1;
			delete []pi_star_1;
		}
		else {
			cout << "File open failed." << endl;
		}
	}
	else {
		cout << ".exe" << endl;
	}
	return 0;
}
