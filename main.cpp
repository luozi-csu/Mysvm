#pragma warning(disable:4996)
#include<vector>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<cmath>
#include<map>
using namespace std;

#define ROW 8124
#define COL 23

vector<vector<double>> dataset(ROW, vector<double>(COL, 0));
vector<vector<double>> trainData, testData;
const char* filePath = "./agaricus-lepiota.data";

#pragma region Init data
void loadData(const char* filePath)
{
	ifstream inFile;
	inFile.open(filePath);
	if (!inFile.is_open()) {
		cout << "can't open the file" << endl;
		exit(0);
	}
	string line;
	int rowCnt = 0;
	char tmp;
	while (rowCnt < ROW) {
		getline(inFile, line);
		stringstream ss;
		ss << line;
		int colCnt = 0;
		while (colCnt < COL) {
			ss >> tmp;
			if (tmp != ',') {
				dataset[rowCnt][colCnt] = (double)tmp - 97;
				++colCnt;
			}
		}
		++rowCnt;
	}
}

void changeLabel()
{
	for (int i = 0; i < ROW; ++i) {
		if (dataset[i][0] == (double)'p' - 97)
			dataset[i][0] = 1;
		else if (dataset[i][0] == (double)'e' - 97)
			dataset[i][0] = -1;
	}
}

void divideData(int p_train, int p_test)
{
	int trainLen = ROW * p_train / (p_train + p_test);
	for (int i = 0; i < trainLen; ++i) {
		vector<double> tmp;
		for (int j = 0; j < COL; ++j)
			tmp.push_back(dataset[i][j]);
		trainData.push_back(tmp);
	}
	for (int i = trainLen; i < ROW; ++i) {
		vector<double> tmp;
		for (int j = 0; j < COL; ++j)
			tmp.push_back(dataset[i][j]);
		testData.push_back(tmp);
	}
}

void initData(const char* filePath, int p_train, int p_test)
{
	loadData(filePath);
	changeLabel();
	divideData(p_train, p_test);
}
#pragma endregion

#pragma region kernel functions
inline double linear(const vector<double>& x1, const vector<double>& x2)
{
	double sum = 0;
	int n = x1.size();
	for (int i = 0; i < n; ++i)
		sum += x1[i] * x2[i];
	return sum;
}

inline double quadratic(const vector<double>& x1, const vector<double>& x2)
{
	double res, sum = 0;
	int n = x1.size();
	for (int i = 0; i < n; ++i)
		sum += x1[i] * x2[i];
	res = sum * sum;
	return res;
}

inline double gaussian(const vector<double>& x1, const vector<double>& x2)
{
	double e = 2.718281828459, sum = 0;
	int n = x1.size();
	for (int i = 0; i < n; ++i)
		sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
	double res = pow(e, -sum);
	return res;
}
#pragma endregion

class SVM
{
public:
	SVM() = default;
	SVM(const vector<vector<double>>& _data, double _C,
		double(*_kf)(const vector<double>& x1, const vector<double>& x2));
	void train(int times, bool epoch) { SMO(times, epoch); }
	double test(const vector<vector<double>>& _testData);
	void setData(const vector<vector<double>>& _data);
	void setC(double _C) { C = _C; }
	void setKf(double(*_kf)(const vector<double>&, const vector<double>&)) { kf = _kf; }
	double testForBagging(const vector<double>& x);
private:
	double(*kf)(const vector<double>& x1, const vector<double>& x2);
	vector<vector<double>> data;
	vector<double> alpha;
	vector<double> omega;
	vector<double> y;
	multimap<int, int> table;
	double b = 0;
	double C;
	bool kkt(int i);
	double calcError(int i);
	pair<int, int> chooseAlphas();
	pair<double, double> calcNewAlphas(int idx1, int idx2);
	void clearOmega();
	void updateOmega();
	void updateTable(int idx1, int idx2);
	double calcB(int idx1, int idx2, double alpha1_new, double alpha2_new);
	void SMO(int times, bool epoch);
};

SVM::SVM(const vector<vector<double>>& _data, double _C,
	double(*_kf)(const vector<double>& x1, const vector<double>& x2) = linear) :
	data(_data), C(_C), kf(_kf)
{
	alpha.resize(data.size(), 0);
	omega.resize(data[0].size() - 1, 0);
	y.resize(data.size(), 0);
	for (int i = 0; i < data.size(); ++i) {
		y[i] = data[i][0];
		table.insert(pair<int, int>(1, i));
	}
}

void SVM::setData(const vector<vector<double>>& _data)
{
	int m = _data.size(), n = _data[0].size();
	data.resize(m, vector<double>(n, 0));
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			data[i][j] = _data[i][j];
	alpha.resize(m, 0);
	omega.resize(n - 1, 0);
	y.resize(m, 0);
	for (int i = 0; i < m; ++i) {
		y[i] = data[i][0];
		table.insert(pair<int, int>(1, i));
	}
}

bool SVM::kkt(int i)
{
	vector<double> x(data[i].begin() + 1, data[i].end());
	double res = y[i] * kf(omega, x) + b - 1;
	if (alpha[i] < 1.0e-4)
		return res >= 1.0e-4;
	else if (abs(alpha[i] - C) < 1.0e-4)
		return res <= 1.0e-4;
	return abs(res) < 1.0e-4;
}

double SVM::calcError(int i)
{
	vector<double> x(data[i].begin() + 1, data[i].end());
	return kf(omega, x) + b - y[i];
}

pair<int, int> SVM::chooseAlphas()
{
	int idx1 = -1, idx2 = -1, n = data.size();
	auto iter = table.begin();
	while (iter != table.end()) {
		int i = iter->second;
		if (kkt(i) == false) {
			idx1 = i;
			double e1 = calcError(idx1);
			double maxDiff = 0;
			for (int j = 0; j < n; ++j) {
				double e2 = calcError(j);
				if (abs(e1 - e2) > maxDiff) {
					maxDiff = abs(e1 - e2);
					idx2 = j;
				}
			}
			break;
		}
		++iter;
	}
	return { idx1, idx2 };
}

pair<double, double> SVM::calcNewAlphas(int idx1, int idx2)
{
	vector<double> x1(data[idx1].begin() + 1, data[idx1].end());
	vector<double> x2(data[idx2].begin() + 1, data[idx2].end());
	double eta = kf(x1, x1) + kf(x2, x2) - 2 * kf(x1, x2);
	double e1 = calcError(idx1), e2 = calcError(idx2);
	double alpha2_uncut = alpha[idx2] + y[idx2] * (e1 - e2) / eta;
	double L, H;
	if (y[idx1] != y[idx2]) {
		L = max((double)0, alpha[idx2] - alpha[idx1]);
		H = min(C, C + alpha[idx2] - alpha[idx1]);
	}
	else {
		L = max((double)0, alpha[idx1] + alpha[idx2] - C);
		H = min(C, alpha[idx1] + alpha[idx2]);
	}
	double alpha2_new;
	if (alpha2_uncut >= H)
		alpha2_new = H;
	else if (alpha2_uncut > L)
		alpha2_new = alpha2_uncut;
	else
		alpha2_new = L;
	double alpha1_new = alpha[idx1] + y[idx1] * y[idx2] * (alpha[idx2] - alpha2_new);
	return { alpha1_new, alpha2_new };
}

void SVM::clearOmega()
{
	for (int i = 0; i < omega.size(); ++i)
		omega[i] = 0;
}

void SVM::updateOmega()
{
	clearOmega();
	int n = data.size();
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < omega.size(); ++j)
			omega[j] += y[i] * alpha[i] * data[i][j + 1];
}

void SVM::updateTable(int idx1, int idx2)
{
	for (auto iter = table.begin(); iter != table.end(); ++iter) {
		if (iter->second == idx1) {
			table.erase(iter);
			break;
		}
	}
	for (auto iter = table.begin(); iter != table.end(); ++iter) {
		if (iter->second == idx2) {
			table.erase(iter);
			break;
		}
	}
	if (alpha[idx1] - 0 > 1.0e-4 && C - alpha[idx1] > 1.0e-4)
		table.insert(pair<int, int>(0, idx1));
	else table.insert(pair<int, int>(1, idx1));
	if (alpha[idx2] - 0 > 1.0e-4 && C - alpha[idx2] > 1.0e-4)
		table.insert(pair<int, int>(0, idx2));
	else table.insert(pair<int, int>(1, idx2));
}

double SVM::calcB(int idx1, int idx2, double alpha1_new, double alpha2_new)
{
	vector<double> x1(data[idx1].begin() + 1, data[idx1].end());
	vector<double> x2(data[idx2].begin() + 1, data[idx2].end());
	double b1 = -calcError(idx1) - y[idx1] * kf(x1, x1) * (alpha1_new - alpha[idx1])
		- y[idx2] * kf(x2, x1) * (alpha2_new - alpha[idx2]) + b;
	double b2 = -calcError(idx2) - y[idx1] * kf(x1, x1) * (alpha1_new - alpha[idx1])
		- y[idx2] * kf(x2, x2) * (alpha2_new - alpha[idx2]) + b;
	double b_new;
	if (alpha1_new > 1.0e-4 && (C - alpha1_new) > 1.0e-4)
		b_new = b1;
	else if (alpha2_new > 1.0e-4 && (C - alpha2_new) > 1.0e-4)
		b_new = b2;
	else b_new = (b1 + b2) / 2;
	return b_new;
}

double SVM::test(const vector<vector<double>>& _testData)
{
	int n = _testData.size(), cnt = 0;
	for (int i = 0; i < n; ++i) {
		vector<double> x(_testData[i].begin() + 1, _testData[i].end());
		double predict = kf(omega, x) + b >= 0 ? 1 : -1;
		if (predict == _testData[i][0])
			++cnt;
	}
	double acc = (double)cnt / (double)n * 100;
	return acc;
}

double SVM::testForBagging(const vector<double>& x)
{
	return kf(omega, x) + b >= 0 ? 1 : -1;
}

void SVM::SMO(int times, bool epoch = true)
{
	int cnt = 0;
	double sum = 0;
	while (cnt < times) {
		pair<int, int> idxes = chooseAlphas();
		int idx1 = idxes.first, idx2 = idxes.second;
		if (idx1 == -1) return;
		pair<double, double> newAlphas = calcNewAlphas(idx1, idx2);
		double alpha1_new = newAlphas.first, alpha2_new = newAlphas.second;
		double b_new = calcB(idx1, idx2, alpha1_new, alpha2_new);
		b = b_new; alpha[idx1] = alpha1_new; alpha[idx2] = alpha2_new;
		updateOmega();
		updateTable(idx1, idx2);
		++cnt;
		double acc = test(testData);
		//if (epoch && cnt % 1 == 0) {
		//	printf("epoch %d accuracy: %.4f\n", cnt, acc);
		//}
		sum += acc;
	}
	sum /= (double)times;
	printf("average accuracy: %.4f\n", sum);
}

vector<vector<vector<double>>> bootstrap(int times)
{
	vector<vector<vector<double>>> sets;
	int n = dataset.size();
	for (int i = 0; i < times; ++i) {
		vector<vector<double>> subset;
		for (int j = 0; j < n; ++j) {
			int num = rand() % n;
			subset.push_back(dataset[num]);
		}
		sets.push_back(subset);
	}
	return sets;
}

void bagging(int times, double _C,
	double (*_kf)(const vector<double>&, const vector<double>&) = linear)
{
	vector<vector<vector<double>>> sets = bootstrap(times);
	SVM* models = new SVM[times];
	for (int i = 0; i < times; ++i) {
		models[i].setData(sets[i]);
		models[i].setC(_C);
		models[i].setKf(_kf);
		printf("model %d \n", i);
		models[i].train(50, true);
	}
	double res = 0;
	for (int i = 0; i < dataset.size(); ++i) {
		vector<double> x(dataset[i].begin() + 1, dataset[i].end());
		double cnt = 0;
		for (int j = 0; j < times; ++j)
			cnt += models[j].testForBagging(x);
		double predict = cnt >= 0 ? 1 : -1;
		if (predict == dataset[i][0])
			++res;
	}
	double acc = res / (double)dataset.size() * 100;
	printf("bagging accuracy: %.4f %\n", acc);
}

int main()
{
	initData(filePath, 2, 1);
	//SVM obj(dataset, 0.009);
	//obj.train(100, true);
	bagging(5, 0.01);
	return 0;
}