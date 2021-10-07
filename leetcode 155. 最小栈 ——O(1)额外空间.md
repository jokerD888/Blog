## leetcode [155. 最小栈](https://leetcode-cn.com/problems/min-stack/) ——O(1)额外空间

### 前言

在力扣管方解答中使用了额外的辅助栈，即用了O(N)的额外空间。

那么如果让你在空间上在优化优化，你会怎么做？

先上代码：

```cpp
//1.若栈为空，特殊考虑，minv=value,push(0)
//2.push时，根据diff=value-minv,插入diff,随后根据的diff的正负，可判断，若diff<0，即说明value小于当前最小值minv,否则，最小值仍为minv
//3.pop时，若栈顶元素diff<0,根据2可知，此次push时minv被修改了，经过minv-=diff即可将minv恢复为其上一个值，否则，minv未被修改不用恢复操作。最后pop即可。
//4.top时，需要把原数据还原，若栈顶元素diff<0,return top即可，否则，return minv + diff，来还原数据进行返回
//5.getMin时，直接返回minv即可，因minv就存储着栈中元素的最小值
//6.数据类型使用 long long 是为了防止val-minv数据溢出
//此法巧妙在：pop时，栈中的值若是负数，那么栈顶元素就是最小值，且最小值会发生变化，要用当前的最小值-栈顶元素来更新最小值。
//若是正数，那么栈顶的值就是当前栈顶值 + 最小值。
class MinStack { 
public:
    stack<long long> minSt;	
    long long minv;		//存储栈中最小值

    MinStack() { }
    
    void push(int val) {
        if (!minSt.size()) {	//栈空，特殊情况
            minSt.push(0);
            minv = val;
        }  else {
            long long diff = val - minv;	//取差值
            minSt.push(diff);		//push差值
            minv = diff < 0 ? val : minv;	//更新minv
        }
    }

    void pop() {
        if (minSt.size()) {
            long long diff = minSt.top();
            minSt.pop();	
            if (diff < 0)
                minv -= diff;	//更新最小值
        }
    }

    int top() {
        long long diff = minSt.top();
        if (diff < 0)
            return minv;
        else
            return minv + diff;		//恢复原数据
    }

    int getMin() {
        return minv;
    }
};
```

再附图解

![](E:\图片\画图\最小栈.png)

