# LAB2 HashFun Report

### **不同哈希策略的实现**

**“坏”哈希函数：**将每个字符的ascii值相加求和后对N取模。
$$
res = {\sum_{i=0}^{strlen(str)}{str[i]}}  \ \%N
$$
**“好”哈希函数：**将字符串根据ascii码值视作128进制数，将其转化为十进制后对N取模，和坏的哈希函数相比，这一函数能区分相同字符通过不同顺序组成的单词，因此更加均匀。
$$
res = {\sum_{i=0}^{strlen(str)}{str[i]}\cdot 128^i}  \ \%N
$$
**双向平方试探策略：**先通过哈希函数为字符串确定一个初始值(`init`中实现)，然后以该位置为轴双向平方试探。在派生类中定义变量成员`stride`记录上一次试探步长，每次试探更新stride，并根据变化前后stride确定新位置。

```c++
int old_stride = stride;
    if (stride <= 0) {	//向右试探
        stride = -stride + 1; 
        int next = ((long long)old_stride * old_stride) % table_size;
        next = (next + (long long)stride * stride) % table_size;
        return (last_choice + next) % table_size;
    } else {	//向左试探
        stride = -stride;
        int next = ((long long)old_stride * old_stride) % table_size;
        next = (next + (long long)stride * stride) % table_size;
        return (last_choice - next + table_size) % table_size;
    }
```

**公共溢出区策略：**

原有`table_size`拿出一半设为缓冲溢出区，在`hashtable`构造函数中用向下转换判断冲突类型

```c++
overflow_probe* find = dynamic_cast<overflow_probe*>(my_collision);	//相下转换，如果成功则是公共溢出策略
if (find) {
    table_size =  size/2;
}
```

每次探测时，从`table_size`（溢出区起点）向后扫描

```c++
if (last_choice < table_size)
    return table_size;
return last_choice + 1;
```

### **测试数据**

**构造方法：**从`poj.txt`中提取所有信息，对其使用`random_shuffle`混排取前`in_num`项。数据生成有三个参数：

- `in_num`：插入的字符串数量，字符串选取方法见上
- `sear_num`：询问次数，其中一半为成功查找，一半为失败查找，失败询问构造方法为`“#_#”+存在的字符串`，正确和失败查找交替出现
- 第三个参数为1时，将插入的键值字符串按字典序升序排序，否则不排序

运行`makedata.bat`脚本可生成对应的数据

**数据特征：**（表大小为200023）

|      | in_num | sear_num | 是否排序 |
| ---- | ------ | -------- | -------- |
| 0    | 10000  | 8000     | 否       |
| 1    | 10000  | 8000     | 是       |
| 2    | 100000 | 80000    | 否       |

- 数据0和数据1相比，数据规模相同（但两次随机取，内容不完全一致），区别在于是否排序
- 数据2和数据0相比，数据规模扩大了10倍

运行`run.bat`进行一组测试，运行的具体结果保存在result0,1,2三个文件夹下。

### **分析结果**

1. 无论规模如何，插入值排序与否，**好哈希都明显提高程序性能**（最好300倍），因为好哈希让散列更均匀，减少冲突次数，排序对运行效率无明显影响。
2. **双向平放和线性试探相比，散列不均匀性能提升明显，均匀时无提升**。可能是不均匀时，双向平方试探可迅速跳出聚集点。而散列均匀时，线性试探也能在附近迅速找到空位。
3. **不均匀散列，开放和封闭效率大体相同，均匀散列，封闭表现明显更优**。另外装填因子较大时，封闭散列明显更占优。公共溢出区策略在处理表的局部比较“满”，再插入少量数据的情况下，可能表现更好。
4. 字符串本身不均匀，可能导致“好”哈希函数生成的散列也不均匀，冲突增多。
5. 可以记录实时的装填因子，对不同哈希策略，实验确定装填因子为多大时扩容或者缩容，使性能最优。
