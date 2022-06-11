# 十三届蓝桥杯省赛C++ B组

## 前言

为巩固知识，也为接下来的比赛做准备，故总体进行复习，学习最优解（也可能还有更优的），汲取养分，故记以此博文。

## 试题 A: 九进制转十进制

```
1478	
```



## 试题 B: 顺子日期

```cpp
#include <iostream>
using namespace std;

// 012 也算顺子
const int months[]{ 0,31,28,31,30,31,30,
					31,31,30,31,30,31 };

bool check(string str) {
	for (int i = 0; i + 2 < str.size(); ++i)
		if (str[i + 1] == str[i] + 1 && str[i + 2] == str[i] + 2)
			return true;

	return false;
}
int main()
{
	int year = 2022, month = 1, day = 1;
	int res = 0;
	for (int i = 0; i < 365; ++i) {
		char str[10];
		sprintf(str, "%04d%02d%02d", year, month, day);
		
		if (check(str))
			++res;

		if (++day > months[month]) {
			day = 1;
			++month;
		}
	}
	cout << res;
	return 0;
}
```

## 试题 C: 刷题统计

```cpp
#include <iostream>
using namespace std;

int main()
{
	long long a, b, n;
	cin >> a >> b >> n;
	// 一周做 5a+2b
	long long w = 5 * a + 2 * b;
	long long res = n / w * 7;	
	n %= w;

	long long d[]{ a,a,a,a,a,b,b };
	for (int i = 0; n > 0; ++i) {
		n -= d[i];
		++res;
	}
	cout << res << endl;
}
```



## 试题 D: 修剪灌木

```cpp
#include <iostream>
using namespace std;

int main()
{
	int n;
	cin >> n;
	for (int i = 1; i <= n; ++i) {
		cout << max(i - 1, n - i) * 2 << endl;	// i位置处的答案为i到两端距离的最大值*2
	}
	return 0;
}
```

## 试题 E: X 进制减法

```cpp
#include <iostream>
using namespace std;
#include <algorithm>

long long mod = 1000000007;
const int MAXN = 1E5 + 10;
int arr[MAXN];	// 存储数Ma
int brr[MAXN];	// 存储数Mb
int main()
{
	int n, m1, m2;
	cin >> n;
	cin >> m1;
	for (int i = m1 - 1; i >= 0; --i)
		cin >> arr[i];
	cin >> m2;
	for (int i = m2 - 1; i >= 0; --i) {
		cin >> brr[i];
	}

	long long res = 0;
	for (int i = m1 - 1; i >= 0; --i) {	//由题A>=B 所以m1>=m2
		res = (res * max({ arr[i] + 1,brr[i] + 1,2 }) + arr[i] - brr[i]) % mod;		//秦九韶算法
	}

	cout << res % mod;

	return 0;
}
```

## 试题 F: 统计子矩阵

解法：二维数组前缀和+枚举边界	时间O(N^4)  AC 70%，下方有最优解。

如果了较二维数组前缀和，此法较易想到，能拿到70%分数。

```cpp
// 二维数组前缀和+暴力枚举起点终点，时间复杂度O(n^4) AC 70%
#include<iostream>
using namespace std;

const int N = 510;
long long sum[N][N];
long long arr[N][N];

void init(int n, int m) {
	for (register int i = 1; i <= n; ++i) {
		for (register int j = 1; j <= m; ++j) {
			cin >> arr[i][j];
			sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + arr[i][j];
		}
	}
}

long long sub(int x1, int y1, int x2, int y2) {
	return sum[x1][y1] - sum[x1][y2] - sum[x2][y1] + sum[x2][y2];
}
int main()
{
	int n, m, k;
	cin >> n >> m >> k;

	init(n, m);
	long long ans = 0;
	// xx > x && yy > y	//直接暴力枚举起点终点，AC 70%	时间复杂度O(n^4)
	for (int xx = 1; xx <= n; ++xx) {
		for (int yy = 1; yy <= m; ++yy) {
			for (int x = 0; x < n; ++x) {
				if (xx <= x)continue;
				for (int y = 0; y < m; ++y) {
					if (yy <= y)continue;
					if (sub(xx, yy, x, y) <= k)
						++ans;
				}
			}
		}
	}
	cout << ans;

	return 0;
}
```

解法：前缀和+枚举左右边界+滑动窗口	时间O(N^3)

```cpp
#include<iostream>
using namespace std;

const int N = 510;
long long sum[N][N];
long long arr[N][N];
// 计算前缀和
void init(int n, int m) {
	for (register int i = 1; i <= n; ++i) {
		for (register int j = 1; j <= m; ++j) {
			cin >> arr[i][j];
			sum[i][j] = sum[i][j - 1] + arr[i][j];
		}
	}
}
// 返回第r行，第c1列到c2列的和
long long sub(int r, int c1, int c2) {
	return sum[r][c2] - sum[r][c1];
}
int main()
{
	int n, m, k;
	cin >> n >> m >> k;

	init(n, m);
	long long ans = 0;

	for (register int l = 0; l < m; ++l) {	// l枚举的是左边那列
		for (register int r = l + 1; r <= m; ++r) {	// r枚举的是右边那列
			// 在 (l,r]这两列中间从上往下玩滑动窗口
			long long total = 0;
			int high = 0, low = 0;
			// 枚举右边界，左边界控制到极限，即右边界固定，左边界再往左移1，和就大于k了
			for (high = 1; high <= n; ++high) {
				total += sub(high, l, r);
				while (total > k) {	// 控制左边界，是左边界和法
					++low;
					total -= sub(low, l, r);
				}
				ans += high - low ;	// 以第high行做矩阵的底边，当前窗口内共有high-low种可能
			}
		}
	}
	cout << ans;

	return 0;
}
```

## 试题 H: 扫雷

```cpp
#include <iostream>
#include <cstring>
using namespace std;
typedef long long LL;
const int N = 50010, M = 999997;	// M是哈希表的大小，一般是数据量的10倍，最低不低于2倍，且利用M作为计算哈希值一般为质数
// 考点：图的遍历DFS/BFS+手写哈希表
int n, m;
struct Circle {
	int x, y, r;
}cir[N];
// 哈希表是通过用户给定的值进行计算得出哈希值，根据哈希值作为哈希表的下标，哈希表中的value存入的就是用户给定的值
LL h[M];	// 哈希表 h[i] 表示i位置表示的key
int id[M];	// id[i] 表示以i作为key,在cir数组中的位置
bool st[M];	// st[i] 表示 i 位置的点访问过没

// 将坐标(x,y)转为为一个long long类型，作为哈希的键
LL get_key(int x, int y) {
	return x * 1000000001LL + y;
}
// 找到（x,y)应该存在哈希表的什么位置
int find(int x, int y) {
	LL key = get_key(x, y);
	int t = (key % M + M) % M;	// 计算哈希值

	while (h[t] != -1 && h[t] != key) {
		if (++t == M)
			t = 0;	// 从头再开始找
	}

	return t;
}
int sqr(int x) {
	return x * x;
}
void dfs(int x, int y, int r) {
	st[find(x, y)] = true;
	// 枚举该坐标周围点，最多20*20的矩阵
	for (int i = x - r; i <= x + r; ++i) {
		for (int j = y - r; j <= y + r; ++j) {
			if (sqr(i - x) + sqr(j - y) <= sqr(r)) {
				int t = find(i, j);
				if (id[t] && !st[t])	
					dfs(i, j, cir[id[t]].r);
			}
		}
	}
}
int main()
{

	scanf("%d%d", &n, &m);
	memset(h, -1, sizeof h);

	for (int i = 1; i <= n; ++i) {
		int x, y, r;
		scanf("%d%d%d", &x, &y,&r);

		cir[i] = { x,y,r };

		int t = find(x, y);	
		if (h[t] == -1)	//如果该位置没存过
			h[t] = get_key(x, y);	// 存入
		if (!id[t] || cir[id[t]].r < r)	// 如果该位置首次记录 或者 该位置的新r大于已有的r
			id[t] = i;			// 更新位置映射
	}

	while (m--) {
		int x, y, r;
		scanf("%d%d%d", &x, &y, &r);
		// 枚举x,y位置的点，最多20*20的范围
		for (int i = x - r; i <= x + r; ++i) {
			for (int j = y - r; j <= y + r; ++j) {
				if (sqr(i - x) + sqr(j - y) <= sqr(r)) {
					int t = find(i, j);
					if (id[t] && !st[t])	// 如果该位置存在 && 没被访问过
						dfs(i, j, cir[id[t]].r);
				}
			}
		}
	}
	int res = 0;
	// 查找哪些雷的坐标被访问过
	for (int i = 1; i <= n; ++i) {
		if (st[find(cir[i].x, cir[i].y)])
			++res;
	}
	printf("%d", res);
	return 0;
}
```

## 试题 I: 李白打酒加强版

```cpp
#include <iostream>
using namespace std;
#include <cstring>
const int N = 110, MOD = 1e9 + 7;

int n, m;
int f[N][N][N];	// f[i][j][k] 表示： 一共遇到i个店，j个花，还有k斗酒的方法数

int main()
{
	cin >> n >> m;
	f[0][0][2] = 1;		// base case
	for (int i = 0; i <= n; ++i)
		for (int j = 0; j <= m; ++j)
			// 遇花喝一斗，所以酒的数量一定不能大于花数
			for (int k = 0; k <= m; ++k) {
				int& v = f[i][j][k];
				if (i && k % 2 == 0)
					v = (v + f[i - 1][j][k / 2]) % MOD;
				if (j)
					v = (v + f[i][j - 1][k + 1]) % MOD;
			}
	// 因题目限制，最后一步一定是遇见了花，只有一斗酒，所以所求即为此
	cout << f[n][m - 1][1] << endl;
	return 0;
}
```

## 试题 J: 砍竹子

优先队列模拟，时间复杂度基本没富裕，O(6NlogN)	6是任何一个高度的竹子不超过6次操作即可变为高度1

```cpp
// 利用优先队列，每次必然先砍最高的，每次砍完后对于连续相同高度的合并一起
#include<iostream>
using namespace std;
#include <cmath>
#include <queue>
typedef long long LL;
const int N = 200010;

int n;
LL h[N];
struct Seg {
	int l, r;
	LL v;
	bool operator<(const Seg& S) const {
		if (v != S.v)
			return v < S.v;		// 高的在堆顶
		return l > S.l;			// 左区间小的在堆顶
	}
};

LL f(LL x)
{
	return sqrt(x / 2 + 1);
}
int main()
{
	scanf("%d", &n);
	for (int i = 0; i < n; ++i) {
		scanf("%lld", &h[i]);
	}
	priority_queue<Seg> heap;
	for (int i = 0; i < n; ++i) {
		// 连续相同的合并在一起，左区间i,右区间i+1开始不断往右扩（只要h[j]==[i])
		int j = i + 1;
		while (j < n && h[i] == h[j])
			++j;	// 右扩
		heap.push({ i,j - 1,h[i] });
		i = j - 1;	// i更新
	}

	int res = 0;
	while (heap.size() > 1 || heap.top().v > 1) {
		auto t = heap.top();
		heap.pop();

		while (heap.size() && heap.top().v == t.v && t.r + 1 == heap.top().l)
		{
			t.r = heap.top().r;	// 更新r
			heap.pop();
		}
		heap.push({ t.l,t.r,f(t.v) });
		if (t.v > 1)
			++res;
	}
	printf("%d\n", res);

	return 0;
}
```

思维

```cpp
#include<iostream>
using namespace std;
#include <cmath>
typedef long long LL;
// 时间O(6N）
// 因无论题目范围哪个数经过sqrt(x/2+1)的操作，最多6次就会变为1
// 所以我们可以枚举每个每个数的每一次操作，一次操作看做一层，若某层中相邻数相同，说明可以一起操作，次数-1
const int N = 200010, M = 10;
int n,m;	// m记录最高操作几次
int f[N][M];	// f[i][j] 表示第i个数的第j层,每经过操作一次，层数-1
int main()
{
	scanf("%d", &n);
	LL stk[M];	// 栈

	int res = 0;
	for (int i = 0; i < n; ++i) {
		// 先单独一个数一个数考虑，
		LL x;
		int top = 0;
		scanf("%lld", &x);
		while (x > 1) stk[++top] = x, x = sqrt(x / 2 + 1);
		res += top;		// 操作累加
		m = max(m, top);	// 记录操作最多次
		
		for (int j = 0, k = top; k; ++j, --k) {
			f[i][j] = stk[k];
		}
	}

	for (int i = 0; i < m; ++i) {	// 最多6层
		for (int j = 1; j < n; ++j) {	// N
			if (f[j][i] && f[j][i] == f[j - 1][i])	// 相邻具有相同高度，可一同操作，res--
				--res;
		}
	}
	printf("%d\n", res);
	return 0;
}
```

## 后言

最后也是记录一下荣获省一的成绩，🤭，但经此发现，自己还有很多不足，还是弱弱一枚，还得精进算法🤤

其中绝大部分来自y总视频讲解，不过自己添加了注释便于理解。[讲解视频见此](https://www.bilibili.com/video/BV18i4y1D7cJ?spm_id_from=333.999.0.0&vd_source=6ae00b4a223ad4d8dbd864c4db6c3bac)

最后给大家几个网络供学习测试,[New Online Judge](http://oj.ecustacm.cn/viewnews.php?id=1021),[C语言网](https://www.dotcpp.com/oj/problemset.php?page=24&mark=6)。
