Greedy Algoritms:
1. Fractional Knapsack

#include <bits/stdc++.h>

using namespace std;

struct Item {
	int value, weight;
	Item(int value, int weight)
	{
		this->value = value;
		this->weight = weight;
	}
};

bool cmp(struct Item a, struct Item b)
{
	double r1 = (double)a.value / (double)a.weight;
	double r2 = (double)b.value / (double)b.weight;
	return r1 > r2;
}

// Main greedy function to solve problem
double fractionalKnapsack(int W, struct Item arr[], int n)
{
		sort(arr, arr + n, cmp);


	double finalvalue = 0.0; 

	// Looping through all Items
	for (int i = 0; i < n; i++) {
		// If adding Item won't overflow, add it completely
		if (arr[i].weight <= W) {
			W -= arr[i].weight;
			finalvalue += arr[i].value;
		}

		// If we can't add current Item, add fractional part
		// of it
		else {
			finalvalue += arr[i].value* ((double)W/ (double)arr[i].weight);
			break;
		}
	}

	// Returning final value
	return finalvalue;
}

// Driver code
int main()
{
	int W = 50; // Weight of knapsack
	Item arr[] = { { 60, 10 }, { 100, 20 }, { 120, 30 } };

	int n = sizeof(arr) / sizeof(arr[0]);

	// Function call
	cout << "Maximum value we can obtain = "
		<< fractionalKnapsack(W, arr, n);
	return 0;
}


2. Activity Selection

#include <bits/stdc++.h>
using namespace std;

void printMaxActivities(int s[], int f[], int n)
{
	int i, j;

	cout <<"Following activities are selected "<< endl;
	i = 0;
	cout <<" "<< i;


	for (j = 1; j < n; j++)
	{
	if (s[j] >= f[i])
	{
		cout <<" " << j;
		i = j;
	}
	}
}

int main()
{
	int s[] = {1, 3, 0, 5, 8, 5};
	int f[] = {2, 4, 6, 7, 9, 9};
	int n = sizeof(s)/sizeof(s[0]);
	printMaxActivities(s, f, n);
	return 0;
}


3. Job Sequencing

#include <algorithm>
#include <iostream>
using namespace std;


struct Job {
	char id; 
	int dead; 
	int profit; 
};

bool comparison(Job a, Job b)
{
	return (a.profit > b.profit);
}

void printJobScheduling(Job arr[], int n)
{
	sort(arr, arr + n, comparison);

	int result[n]; 
	bool slot[n]; 

	
	for (int i = 0; i < n; i++)
		slot[i] = false;

	
	for (int i = 0; i < n; i++) {
		
		for (int j = min(n, arr[i].dead) - 1; j >= 0; j--) {
		
			if (slot[j] == false) {
				result[j] = i; 
				slot[j] = true; 
				break;
			}
		}
	}

	for (int i = 0; i < n; i++)
		if (slot[i])
			cout << arr[result[i]].id << " ";
}


int main()
{
	Job arr[] = { { 'a', 2, 100 },
				{ 'b', 1, 19 },
				{ 'c', 2, 27 },
				{ 'd', 1, 25 },
				{ 'e', 3, 15 } };
	int n = sizeof(arr) / sizeof(arr[0]);
	cout << "Following is maximum profit sequence of jobs "
			"\n";

	// Function call
	printJobScheduling(arr, n);
	return 0;
}


4 Eulerian Path and Circuit
#include<iostream>
#include<vector>
#define NODE 5
using namespace std;

int graph[NODE][NODE] = {
   {0, 1, 1, 1, 0},
   {1, 0, 1, 0, 0},
   {1, 1, 0, 0, 0},
   {1, 0, 0, 0, 1},
   {0, 0, 0, 1, 0}
};
                               
/* int graph[NODE][NODE] = {
   {0, 1, 1, 1, 1},
   {1, 0, 1, 0, 0},
   {1, 1, 0, 0, 0},
   {1, 0, 0, 0, 1},
   {1, 0, 0, 1, 0}
};
*/    //uncomment to check Euler Circuit
                               
/* int graph[NODE][NODE] = {
   {0, 1, 1, 1, 0},
   {1, 0, 1, 1, 0},
   {1, 1, 0, 0, 0},
   {1, 1, 0, 0, 1},
   {0, 0, 0, 1, 0}
};
*/    //Uncomment to check Non Eulerian Graph
               
void traverse(int u, bool visited[]) {
   visited[u] = true;    //mark v as visited

   for(int v = 0; v<NODE; v++) {
      if(graph[u][v]) {
         if(!visited[v])
            traverse(v, visited);
      }
   }
}

bool isConnected() {
   bool *vis = new bool[NODE];
   //for all vertex u as start point, check whether all nodes are visible or not
   for(int u; u < NODE; u++) {
      for(int i = 0; i<NODE; i++)
         vis[i] = false;    //initialize as no node is visited
               
      traverse(u, vis);
         
      for(int i = 0; i<NODE; i++) {
         if(!vis[i])    //if there is a node, not visited by traversal, graph is not connected
            return false;
      }
   }
   return true;
}

int isEulerian() {
   if(isConnected() == false)    //when graph is not connected
      return 0;
   vector<int> degree(NODE, 0);
   int oddDegree = 0;

   for(int i = 0; i<NODE; i++) {
      for(int j = 0; j<NODE; j++) {
         if(graph[i][j])
            degree[i]++;    //increase degree, when connected edge found
      }

      if(degree[i] % 2 != 0)    //when degree of vertices are odd
         oddDegree++; //count odd degree vertices
   }

   if(oddDegree > 2)    //when vertices with odd degree greater than 2
      return 0;
         
   return (oddDegree)?1:2;    //when oddDegree is 0, it is Euler circuit, and when 2, it is Euler path
}

int main() {
   int check;
   check = isEulerian();

   switch(check) {
      case 0: cout << "The graph is not an Eulerian graph.";
         break;
      case 1: cout << "The graph has an Eulerian path.";
         break;
      case 2: cout << "The graph has a Eulerian circuit.";
         break;
   }
}

Topological sort


#include<bits/stdc++.h>
using namespace std;
const int maxn = 1005;
vector<int>graph[maxn];
priority_queue<int, vector<int>, greater<int> > PQ;
int in_degree[maxn];
vector<int> ans;

int main()
{
    int n,m;
    cin >> n >> m;
    for(int i = 0; i < m; i++)
    {
        int u,v;
        cin >> u >> v;
        graph[u].push_back(v);
        in_degree[v]++;
    }

    for(int i = 1; i <= n; i++)
    {
        //  cout << in_degree[i] << " ";
        if(in_degree[i] == 0)
            PQ.push(i);
    }
    if(PQ.size() == 0)
    {
        cout << "Cycle Detected\n";
        return 0;
    }

    //cout << PQ.size()<<endl;
//    for(int i = 0; i < n; i++){
//        for(int x = 0; x < graph[i].size(); x++)
//            cout << graph[i][x]<< " ";
//        cout << "-----------\n";;
//    }
    while(!PQ.empty())
    {
        int u = PQ.top();
        // cout <<  u << endl;
        ans.push_back(u);

        PQ.pop();
        for(int x = 0; x < graph[u].size(); x++)
        {
            int v = graph[u][x];
          //  cout << u << " "<<v << endl;
            in_degree[v]--;
            if(in_degree[v] == 0)PQ.push(v);
        }
    }
    //cout << ans.size()<<endl;
    int sz = ans.size();
     if(sz < n)
    {
        cout << "Cycle Detected\n";
        return 0;
    }
    for(int i = 0; i < sz; i++)
    {
        cout << ans[i]<< " ";
    }
    return 0;
}




LCS
/* A Top-Down DP implementation of LCS problem */
#include <bits/stdc++.h>
using namespace std;

/* Returns length of LCS for X[0..m-1], Y[0..n-1] */
int lcs(char* X, char* Y, int m, int n,
		vector<vector<int> >& dp)
{
	if (m == 0 || n == 0)
		return 0;
	if (X[m - 1] == Y[n - 1])
		return dp[m][n] = 1 + lcs(X, Y, m - 1, n - 1, dp);

	if (dp[m][n] != -1) {
		return dp[m][n];  
	}
	return dp[m][n] = max(lcs(X, Y, m, n - 1, dp),
						lcs(X, Y, m - 1, n, dp));
}

/* Driver code */
int main()
{
	char X[] = "AGGTAB";
	char Y[] = "GXTXAYB";

	int m = strlen(X);
	int n = strlen(Y);
	vector<vector<int> > dp(m + 1, vector<int>(n + 1, -1));
	cout << "Length of LCS is " << lcs(X, Y, m, n, dp);

	return 0;
}
MCM
// C++ program using memoization
#include<bits/stdc++.h>

 
// Matrix Ai has dimension p[i-1] x p[i] for i = 1..n
int MatrixChainOrder(int p[], int n)
{
 
    /* For simplicity of the program, one extra row and one extra column are
       allocated in m[][].  0th row and 0th column of m[][] are not used */
    int m[n][n];
 
    int i, j, k, L, q;
 
    /* m[i,j] = Minimum number of scalar multiplications needed to compute
       the matrix A[i]A[i+1]...A[j] = A[i..j] where dimention of A[i] is
       p[i-1] x p[i] */
 
    // cost is zero when multiplying one matrix.
    for (i = 1; i < n; i++)
        m[i][i] = 0;
 
    // L is chain length.  
    for (L=2; L<n; L++)   
    {
        for (i=1; i<=n-L+1; i++)
        {
            j = i+L-1;
            m[i][j] = INT_MAX;
            for (k=i; k<=j-1; k++)
            {
                // q = cost/scalar multiplications
                q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j];
                if (q < m[i][j])
                    m[i][j] = q;
            }
        }
    }
 
    return m[1][n-1];
}
 
int main()
{
    int arr[] = {1, 2, 3, 4};
    int size = sizeof(arr)/sizeof(arr[0]);
 
    printf("Minimum number of multiplications is %d ",
                       MatrixChainOrder(arr, size));
 
    getchar();
    return 0;
}
 
0/1 knapsack woth sequence 

#include <bits/stdc++.h>
#include <iostream>
using namespace std;


int max(int a, int b) { return (a > b) ? a : b; }

void printknapSack(int W, int wt[], int val[], int n)
{
	int i, w;
for (i = 0; i <= n; i++) {
		for (w = 0; w <= W; w++) {
			if (i == 0 || w == 0)
				K[i][w] = 0;
			else if (wt[i - 1] <= w)
				K[i][w] = max(val[i -1] +
					K[i - 1][w - wt[i - 1]], K[i - 1][w]);
			else
				K[i][w] = K[i - 1][w];
		}
	}

	
	int res = K[n][W];
	cout<< res << endl;
	
	w = W;
	for (i = n; i > 0 && res > 0; i--) {
		
		if (res == K[i - 1][w])
			continue;
		else {
			cout<<" "<<wt[i - 1] ;
			res = res - val[i - 1];
			w = w - wt[i - 1];
		}
	}
}

int main()
{
	int val[] = { 60, 100, 120 };
	int wt[] = { 10, 20, 30 };
	int W = 50;
	int n = sizeof(val) / sizeof(val[0]);
	
	printknapSack(W, wt, val, n);
	
	return 0;
}

Optimal BST

#include <bits/stdc++.h>
using namespace std;

int sum(int freq[], int i, int j);

int optimalSearchTree(int keys[], int freq[], int n)
{
	
	int cost[n][n];
	for (int i = 0; i < n; i++)
		cost[i][i] = freq[i];


	for (int L = 2; L <= n; L++)
	{
	
		for (int i = 0; i <= n-L+1; i++)
		{
			
			int j = i+L-1;
			cost[i][j] = INT_MAX;
			for (int r = i; r <= j; r++)
			{
			int c = ((r > i)? cost[i][r-1]:0) +
					((r < j)? cost[r+1][j]:0) +
					sum(freq, i, j);
			if (c < cost[i][j])
				cost[i][j] = c;
			}
		}
	}
	return cost[0][n-1];
}

int sum(int freq[], int i, int j)
{
	int s = 0;
	for (int k = i; k <= j; k++)
	s += freq[k];
	return s;
}

int main()
{
	int keys[] = {10, 12, 20};
	int freq[] = {34, 8, 50};
	int n = sizeof(keys)/sizeof(keys[0]);
	cout << "Cost of Optimal BST is " << optimalSearchTree(keys, freq, n);
	return 0;
}


lcs with all sequences 

	int lcs(string x,string y){
		int m = x.size(),n = y.size();
		int dp[m+1][n+1];
		for(int i=0;i<=m;i++){
			dp[i][0] = 0;
		}
		for(int j=0;j<=m;j++){
			dp[0][j] = 0;
		}
		for(int i=1;i<=m;i++){
			for(int j=1;j<=n;j++){
				if(x[i-1] == y[j-1]){
					dp[i][j] = dp[i-1][j-1]+1;
				}
				else{
					dp[i][j] = max(dp[i][j-1],dp[i-1][j]);
				}
			}
		}
		return dp[m][n];
	}

Coin change 
#include <bits/stdc++.h>
using namespace std;

int main() {
	
    int N, M;
    cin>>N;
    cin>>M;
    int *coin = new int[M];
    long *change = new long[N+1];
    for(int i = 0; i < M; i++) {
    	cin>>coin[i];
    }

    memset(change, 0, sizeof(change));

    // Base case: There is 1 way to make change for zero cents, use no coins
    change[0] = 1;

    for(int i = 0; i < M; i++) {
        for(int j = coin[i]; j <= N; j++) {
            change[j] += change[j-coin[i]];
        }
    }
    
    // Print solution
    cout<<change[N];
    return 0;
}


Iterative Merge sort

#include<bits/stdc++.h>
using namespace std;

void merge(int arr[], int l, int m, int r);

void mergeSort(int arr[], int l, int r)
{
	if (l < r)
	{
		
		// Same as (l+r)/2 but avoids
		// overflow for large l & h
		int m = l + (r - l) / 2;
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);
		merge(arr, l, m, r);
	}
}

// Function to merge the two haves arr[l..m]
// and arr[m+1..r] of array arr[]
void merge(int arr[], int l, int m, int r)
{
	int k;
	int n1 = m - l + 1;
	int n2 = r - m;

	// Create temp arrays
	int L[n1], R[n2];

	// Copy data to temp arrays L[] and R[]
	for(int i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for(int j = 0; j < n2; j++)
		R[j] = arr[m + 1+ j];

	// Merge the temp arrays
	// back into arr[l..r]
	int i = 0;
	int j = 0;
	k = l;
	
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			j++;
		}
		k++;
	}

	// Copy the remaining elements
	// of L[], if there are any
	while (i < n1)
	{
		arr[k] = L[i];
		i++;
		k++;
	}

	// Copy the remaining elements
	// of R[], if there are any
	while (j < n2)
	{
		arr[k] = R[j];
		j++;
		k++;
	}
}

// Function to print an array
void printArray(int A[], int size)
{
	for(int i = 0; i < size; i++)
		printf("%d ", A[i]);
		
	cout << "\n";
}

// Driver code
int main()
{
	int arr[] = { 12, 11, 13, 5, 6, 7 };
	int arr_size = sizeof(arr) / sizeof(arr[0]);

	cout << "Given array is \n";
	printArray(arr, arr_size);

	mergeSort(arr, 0, arr_size - 1);

	cout << "\nSorted array is \n";
	printArray(arr, arr_size);
	return 0;
}

// This code is contributed by Mayank Tyagi

Recursive Merge sort
// C++ program for Merge Sort
#include <iostream>
using namespace std;

// Merges two subarrays of array[].
// First subarray is arr[begin..mid]
// Second subarray is arr[mid+1..end]
void merge(int array[], int const left, int const mid, int const right)
{
	int  a1 = mid - left + 1;
	int  a2 = right - mid;

	// Create temp arrays
	auto *lftarr = new int[a1],
		*rightarr = new int[a2];

	// Copy data to temp arrays lftarr[] and rightarr[]
	for (auto i = 0; i < a1; i++)
		lftarr[i] = array[left + i];
	for (auto j = 0; j < a2; j++)
		rightarr[j] = array[mid + 1 + j];

	auto indexOfa1 = 0, // Initial index of first sub-array
		indexOfa2 = 0; // Initial index of second sub-array
	int indexOfMergedArray = left; // Initial index of merged array

	// Merge the temp arrays back into array[left..right]
	while (indexOfa1 < a1 && indexOfa2 < a2) {
		if (lftarr[indexOfa1] <= rightarr[indexOfa2]) {
			array[indexOfMergedArray] = lftarr[indexOfa1];
			indexOfa1++;
		}
		else {
			array[indexOfMergedArray] = rightarr[indexOfa2];
			indexOfa2++;
		}
		indexOfMergedArray++;
	}
	// Copy the remaining elements of
	// left[], if there are any
	while (indexOfa1 < a1) {
		array[indexOfMergedArray] = lftarr[indexOfa1];
		indexOfa1++;
		indexOfMergedArray++;
	}
	// Copy the remaining elements of
	// right[], if there are any
	while (indexOfa2 < a2) {
		array[indexOfMergedArray] = rightarr[indexOfa2];
		indexOfa2++;
		indexOfMergedArray++;
	}
}

// begin is for left index and end is
// right index of the sub-array
// of arr to be sorted */
void mergeSort(int array[], int const begin, int const end)
{
	if (begin < end)
{

	auto mid = begin + (end - begin) / 2;
	mergeSort(array, begin, mid);
	mergeSort(array, mid + 1, end);
	merge(array, begin, mid, end);
}
}

void printArray(int A[], int size)
{
	for (auto i = 0; i < size; i++)
		cout << A[i] << " ";
}


int main()
{
	int arr[] = { 12, 11, 13, 5, 6, 7 };
	auto arr_size = sizeof(arr) / sizeof(arr[0]);

	cout << "Given array is \n";
	printArray(arr, arr_size);

	mergeSort(arr, 0, arr_size - 1);

	cout << "\nSorted array is \n";
	printArray(arr, arr_size);
	return 0;
}


Quick Sort

#include <bits/stdc++.h>
using namespace std;


void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

int partition (int arr[], int low, int high)
{
	int pivot = arr[high]; 
	int i = (low - 1); 

	for (int j = low; j <= high - 1; j++)
	{
		
		if (arr[j] < pivot)
		{
			i++;
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}

void quickSort(int arr[], int low, int high)
{
	if (low < high)
	{

		int pi = partition(arr, low, high);

		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
}

void printArray(int arr[], int size)
{
	int i;
	for (i = 0; i < size; i++)
		cout << arr[i] << " ";
	cout << endl;
}

int main()
{
	int arr[] = {10, 7, 8, 9, 1, 5};
	int n = sizeof(arr) / sizeof(arr[0]);
	quickSort(arr, 0, n - 1);
	cout << "Sorted array: \n";
	printArray(arr, n);
	return 0;
}

Binary Search
#include <stdio.h>
 
// Iterative implementation of the binary search algorithm to return
// the position of `target` in array `nums` of size `n`
int binarySearch(int nums[], int n, int target)
{
    // search space is nums[low…high]
    int low = 0, high = n - 1;
 
    // loop till the search space is exhausted
    while (low <= high)
    {
        // find the mid-value in the search space and
        // compares it with the target
 
        int mid = (low + high)/2;    // overflow can happen
        // int mid = low + (high - low)/2;
        // int mid = high - (high - low)/2;
 
        // target value is found
        if (target == nums[mid]) {
            return mid;
        }
 
        // if the target is less than the middle element, discard all elements
        // in the right search space, including the middle element
        else if (target < nums[mid]) {
            high = mid - 1;
        }
 
        // if the target is more than the middle element, discard all elements
        // in the left search space, including the middle element
        else {
            low = mid + 1;
        }
    }
 
    // target doesn't exist in the array
    return -1;
}
 
int main()
{
    int nums[] = { 2, 5, 6, 8, 9, 10 };
    int target = 5;
 
    int n = sizeof(nums)/sizeof(nums[0]);
    int index = binarySearch(nums, n, target);
 
    if (index != -1) {
        printf("Element found at index %d", index);
    }
    else {
        printf("Element not found in the array");
    }
 
    return 0;
}
Recursive Binary Search
#include <stdio.h>
 
// Recursive implementation of the binary search algorithm to return
// the position of `target` in subarray nums[low…high]
int binarySearch(int nums[], int low, int high, int target)
{
    // Base condition (search space is exhausted)
    if (low > high) {
        return -1;
    }
 
    // find the mid-value in the search space and
    // compares it with the target
 
    int mid = (low + high)/2;    // overflow can happen
    // int mid = low + (high - low)/2;
 
    // Base condition (target value is found)
    if (target == nums[mid]) {
        return mid;
    }
 
    // discard all elements in the right search space,
    // including the middle element
    else if (target < nums[mid]) {
        return binarySearch(nums, low, mid - 1, target);
    }
 
    // discard all elements in the left search space,
    // including the middle element
    else {
        return binarySearch(nums, mid + 1, high, target);
    }
}
 
int main(void)
{
    int nums[] = { 2, 5, 6, 8, 9, 10 };
    int target = 5;
 
    int n = sizeof(nums)/sizeof(nums[0]);
 
    int low = 0, high = n - 1;
    int index = binarySearch(nums, low, high, target);
 
    if (index != -1) {
        printf("Element found at index %d", index);
    }
    else {
        printf("Element not found in the array");
    }
 
    return 0;
}

