# **AcWing 109. 天才ACM**

#### 题目描述

[点这里做题](https://www.acwing.com/problem/content/111/)

给定一个整数 MM，对于任意一个整数集合 SS，定义“校验值”如下:

从集合 SS 中取出 MM 对数(即 2×M2×M 个数，不能重复使用集合中的数，如果 SS 中的整数不够 MM 对，则取到不能取为止)，使得“每对数的差的平方”之和最大，这个最大值就称为集合 SS 的“校验值”。

现在给定一个长度为 NN 的数列 AA 以及一个整数 TT。

我们要把 AA 分成若干段，使得每一段的“校验值”都不超过 TT。

求最少需要分成几段。

#### 输入格式

第一行输入整数 KK，代表有 KK 组测试数据。

对于每组测试数据，第一行包含三个整数 N,M,TN,M,T 。

第二行包含 NN 个整数，表示数列A1,A2…ANA1,A2…AN。

#### 输出格式

对于每组测试数据，输出其答案，每个答案占一行。

#### 数据范围

1≤K≤121≤K≤12,
1≤N,M≤5000001≤N,M≤500000,
0≤T≤10180≤T≤1018,
0≤Ai≤2200≤Ai≤220

#### 输入样例：

```
2
5 1 49
8 2 1 7 9
5 1 64
8 2 1 7 9
```

#### 输出样例：

```
2
1
```

### 法一：贪心+二分 O(n^2logn)

```cpp
// 贪心：关于求区间[l,r]的校验值，排序（nlogn)后再从两端向中间逐一配对
// 再利用二分求每个最大区间的右边界logn
// 考虑最坏情况，T 极小，每次 L 只 +2，导致二分 n 次,实际上，时间复杂度最坏为 O(n^2logn)

#include <iostream>
#include <algorithm>
using namespace std;

typedef long long ll;
const int N = 500005;
int n, m;
int ans;              // 存答案
ll T;                 // 题目中的 T
ll w[N], t[N];        // w 是输入的数组，t 是用于求校验值的数组

ll sq(ll x) { return x * x; }

ll get(int l, int r)  // 求原数组区间 [l, r] 的校验值
{
	for (int i = l; i <= r; i++)  t[i] = w[i];
	sort(t + l, t + r + 1);   // 将复制过来的数排序，注意排序区间左闭右开！！
	ll sum = 0;       // 存返回的校验值
	int half = r - l + 1 >> 1;	// 个数的一半
	for (int i = 0; i < m && i < half; ++i) sum += sq(t[l + i] - t[r - i]);
	return sum;
}

int main()
{
	int K;            // 测试数据组数
	cin >> K;
	while (K--)
	{
		cin >> n >> m >> T;
		for (int i = 0; i < n; i++) cin >> w[i];
		ans = 0;      // 答案归零
		int start = 0;    // start 记录当前剩余的区间左端点
		while (start < n) // start < n 说明当前数组还有值，需要继续划分。结束时 start 应等于 n 
		{
			// 当T较小，每次L只+2，导致二分n次，而下面的操作时间为O(nlogn),故总体O(n^2logn)
			int l = start, r = n - 1, mid; // 二分求出当前能划分的最长的区间
			while (l <= r) // 二分查找大于T的位置
			{
				mid = l + r >> 1;
				if (T < get(start, mid)) r = mid - 1;
				else    l = mid + 1;
			}
			start = l;    // 二分完后，l 即当前可划分的最长区间的下一个位置，将 start 制为 l。
			ans++;      // 每次划分完一个区间，ans ++ 
		}
		printf("%d\n", ans);
	}
	return 0;
}
```

### 法二：倍增  O(nlog^2n)

```cpp
// 法一中在较好的状态下可以做到O(nlog^2n)，但若所给T非常小，导致分成的段数也非常多
// 而通过倍增，每次寻找区间的时间复杂度始终为区间长度的对数
// 设答案中每段区间的长度分别为 len1,len2,...,lenk对于每个 leni，需要倍增 O(logn) 次
// 这样每次找到一段，需要 O(logn*leni*logleni) 的时间
// 总时间复杂度即为 O((len1loglen1 + len2loglen2 + ⋯ + lenk)logn) = O(nlog^2n)
// 倍增相比于二分，优化的是寻找每段区间右端点的位置，二分的话，一上来得get(start,(n-start)/2),即O(nlogn)的时间
#include <iostream>
#include <algorithm>
using namespace std;

typedef long long ll;
const int N = 500005;

int n, m;
int ans;
ll T;
ll w[N], t[N];

ll sq(ll x) { return x * x; }

ll get(int l, int r)  // 求原数组区间 [l, r] 的校验值
{
	for (int i = l; i <= r; i++)  t[i] = w[i];
	sort(t + l, t + r + 1);   // 将复制过来的数排序，注意排序区间左闭右开！！
	ll sum = 0;       // 存返回的校验值
	int half = r - l + 1 >> 1;	// 个数的一半
	for (int i = 0; i < m && i < half; ++i) sum += sq(t[l + i] - t[r - i]);
	return sum;
}

int main()
{
	int K;
	scanf("%d", &K);
	while (K--)
	{
		scanf("%d %d %lld\n", &n, &m, &T); // 不用 scanf 的话，这么做有一个点会 TLE
		for (int i = 0; i < n; i++)
			scanf("%lld", &w[i]);
		ans = 0;
		int start = 0, end = 0;            // start 记录剩余区间开头节点，end 记录当前考虑区间的尾结点（左闭右开）
		while (end < n)
		{
			int len = 1;                   // len为步长 初始化为 1
			while (len)                    // len 为 0 自动跳出
			{
				if (end + len < n && get(start, end + len) <= T) // 如果说 len + end 还在 n 以内，且区间 [start, end + len] 的校验值不大于 T
					end += len, len <<= 1; // 那么 end += len，len *= 2
				else    len >>= 1;         // 否则 len /= 2
			}
			start = ++end;               // 让 start 指向当前区间末尾结点的下一个位置
			ans++;                       // 每次循环都找到了一个区间，所以让 ans ++ 
		}
		printf("%d\n", ans);
	}
	return 0;
}
```

### 法三：倍增 + 归并 O(nlogn)

```cpp
// 法二中的瓶颈在于排序，我们可以发现在倍增中start是不变的，只有end是变化的，但是我们每次都是对[start,end]进行排序，
// 显然前面有一部分已然是有序的。即在处理 [start,end)的时候，已经将 [start,end) 排好序了，所以不需要在处理 [start,end+len) 时再排序。
// 处理[start, end + len) 时，只需要将[end, end + len) 排序，然后将[start, end) 与[end, end + len) 这两段区间进行归并即可。

// 假设一共将数组划分成了 k 个区间（这里的区间指的是倍增增加的区间），
// 每个区间的长度分别为 len1,len2,⋯,lenklen1,len2,⋯,lenk。
// 那么按上述方法只需要将每个区间排序一遍，所以时间复杂度为 O(len1loglen1 + len2loglen2 + ⋯ + lenkloglenk)≤O(nlogn))
// 加上每次归并的时间复杂度为 O(n)总的时间复杂度为 O(n + nlogn) = O(nlogn)

#include <iostream>
#include <algorithm>
using namespace std;

typedef long long ll;
const int N = 500005;

int n, m;
int ans;
ll T;
ll w[N], t[N];
ll tmp[N];

ll sq(ll x) { return x * x; }

ll get(int l, int mid, int r)	// 计算[l,r]校验和，[l,mid]已有序，[mid+1,r]待排
{
	for (int i = mid + 1; i <= r; i++) t[i] = w[i]; //复制[mid + 1, r]
	sort(t + mid + 1, t + r + 1);
	int i = l, j = mid + 1, k;              // 双指针进行区间合并
	for (k = l; i <= mid && j <= r; ++k)
		tmp[k] = t[i] < t[j] ? t[i++] : t[j++];

	while (i <= mid) tmp[k++] = t[i++]; // 处理剩下的元素
	while (j <= r) tmp[k++] = t[j++];

	ll sum = 0;                             // 计算校验值
	int half = r - l + 1 >> 1;	// 个数的一半
	for (int i = 0; i < m && i < half; ++i) sum += sq(tmp[l + i] - tmp[r - i]);
	return sum;                        // 返回当前区间 [l, r) 是否合法
}

int main()
{
	int K;
	scanf("%d", &K);
	while (K--)
	{
		scanf("%d %d %lld\n", &n, &m, &T);
		for (int i = 0; i < n; i++)
			scanf("%lld", &w[i]);
		ans = 0;
		int start = 0, end = 0, len;
		t[0] = w[0];	// 第一个get(0,0,1),排序的是[1,1]，第0个不会被选中，手动复制过去
		while (end < n)
		{
			len = 1;
			while (len)
			{
				if (end + len < n && get(start, end, end + len) <= T) // 如果 w 的 [start, end + len] 区间合法
				{
					end += len, len <<= 1;
					for (int i = start; i <= end; ++i)t[i] = tmp[i];	// 新排完序的再复制回来
				} else    len >>= 1;
			}
			start = ++end;	// 头指点上一个段的右端点的下一个
			ans++;
		}
		printf("%d\n", ans);
	}
	return 0;
}

```

