# exercise_book
一些经典算法题的记录

## 经典算法
>快速排序
```python
def quickSort(nums):
    if not nums or len(nums) == 1:
        return nums
    mid = nums[0]
    right = [i for i in nums[1:] if i >= mid]
    left = [i for i in nums[1:] if i <mid]
    return quickSort(left) + [mid] + quickSort(right)

print quickSort([11,3,3,4,2,1,-9])
```

----------
> 归并排序
```python
def merge(a, b):
    c = []
    i=j=0
    while i<len(a) and j<len(b):
        if a[i] < b[j]:
            c.append(a[i])
            i+=1
        else:
            c.append(b[j])
            j+=1
    if i==len(a):
        c+=b[j:]
    else:
        c+=a[i:]
    return c

def mergeSort(nums):
    if len(nums) <= 0:
        return nums
    middle = len(nums)/2
    left = mergeSort(nums[:middle])
    right = mergeSort[nums[middle:]]
    return merge(left, right)
```

## 错题记录
> **二叉树的重建**：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
```
解析：主要用到分治法的思想。前序遍历可以确定每个子树根节点的位置，在中序遍历的序列中找到对应的节点，其左右两边的子序列，即为该节点的左右子树
```
```
class Solution:
    # pre，tin分别为输入的前序遍历的列表和中序遍历的列表
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre:
            return None
        if len(pre) == 1:
            return TreeNode(pre[0])
        else:
            root = TreeNode(pre[0])
            root.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1], tin[:tin.index(pre[0])])
            root.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:], tin[tin.index(pre[0])+1:])
            return root
```

----------
>**0-1背包问题**
```
解析：解决该问题的主要方法是动态规划，有递归法和非递归两种方法
```
```
# 递归，记忆体化方式解决0-1背包问题
# 装饰器的主要作用是直接返回递归中已经执行过的函数的结果
from functools32 import Iru_cache
#w, v, c分别为物品重量，价值，背包容量
def rec_knapsack(w, v, c):
    @Iru_cache
    def m(k, r):
        if k == 0 or r == 0:return 0
        i = k - 1
        drop = m(k-1, r)
        if w[i] > r: return drop
        return max(drop, v[i] + m(k-1, r-w[i]))
    return m(len(m), c)
```
```
# 迭代
def knapsack(w, v, c):
    n = len(w)
    m = [[0]*(c+1) for i in range(n+1)]
    # 当没有物品或者背包容量为0时，最大价格肯定为0，因此从1开始循环 
    for i in range(1,n+1):
        for j in range(1,c+1):
            if w[i-1] <= j:
                m[i][j] = max(m[i-1][j], m[i-1][j-w[i-1]]+v[i-1])
            else:
                m[i][j] = m[i-1][j]
    return m[-1][-1]
```

----------
>**二叉树的递归问题**
```
解析：在二叉树的算法当中最基础的就是层次遍历和前、中、后序遍历，对于很多二叉树的问题来说，分治法是最主要的解决方法
```
```
# 路径最深问题，求一颗二叉树的最深路径
def maxDepthTree(root):
    if not root:
        return 0
    return max(1+maxDepthTree(root.left), 1+maxDepthTree(root.right))
```
```
# 根节点到叶子结点的路径总和问题，给定路径总和，求是否有这么一条路径
def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return False
        if not root.right and not root.left:
            return sum == root.val
        if not root.right and root.left:
            return hasPathSum(root.left, sum-root.val)
        if not root.left and root.right:
            return hasPathSum(root.right, sum-root.val)
        return hasPathSum(root.left, sum-root.val) or hasPathSum(root.right, sum-root.val)
```
```
# 最近公共祖先问题
def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or root == q or root == p:
            return root
        left = lowestCommonAncestor(root.left, p, q)
        right = lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        if left and not right:
            return left
        if right and not left:
            return right
        return None
```
```
# 输入两个二叉树A，B，判断B是否为A的子树
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        result = False
        if pRoot1.val == pRoot2.val:
            result = self.isSubtree(pRoot1, pRoot2)
        if not result:
            result =  self.HasSubtree(pRoot1.right, pRoot2) | self.HasSubtree(pRoot1.left, pRoot2)
        return result
    def isSubtree(self, root1, root2):
        if not root2:
            return True
        if not root1:
            return False
        if root1.val == root2.val:
            return self.isSubtree(root1.left, root2.left) & self.isSubtree(root1.right, root2.right)
        return False
```
