//k-means.cpp
#include <iostream>
#include <cstring>
#include <math.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// create Point Struct
typedef struct
{
    float x;
    float y;
    int label;
} Point;

// Print point
void printPt(Point A)
{
    printf("x\ty\tlabel\n");
    printf("%.2f\t%.2f\t%d\n", A.x, A.y, A.label);
}

// Print point
void printPt(Point* A, int n_data)
{
    printf("x\ty\tlabel\n");
    for(int i = 0; i < n_data; ++i)
        printf("%.2f\t%.2f\t%d\n", A[i].x, A[i].y, A[i].label);
}

// labelling
void labelPoint(Point &data, int label)
{
    data.label = label;
}

// calculate Euclidean Distance
float dist(Point p1, Point p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// calculate means
float calcMean(float* data, int n_data)
{
    float sum = 0;
    for(int i; i < n_data; ++i)
    {
        sum+=data[i];
    }
    return sum/(float)n_data;
}

void bubbleSort(float* data, int* label, int n_data)
{
	// do iteration
	bool any_swap = true;
	float temp;
	int i = 0;
	while(true)
	{
		any_swap = false;
		for(i = 0;i < n_data-1; ++i)
		{
			if(data[i] > data[i+1])
			{
				temp = data[i];
				data[i] = data[i+1];
				data[i+1] = temp;
				temp = label[i];
				label[i] = label[i+1];
				label[i+1] = temp;
				any_swap = true;
			}
		}

		if(!any_swap)
			break;
	}
}

float absf(float x)
{
    if(x < 0)
        return x*-1;
    else
        return x;
}

// calculate K-Means
void kmeans(int k, Point* data, int n_data, Point* (&centroids))
{
    // 2. calculate all distance between centroid and all samples
    float *distData = (float*)malloc(k*sizeof(float));
    int *distIdx = (int*)malloc(k*sizeof(int));
    float *sumX = (float*)malloc(k*sizeof(float));
    float *sumY = (float*)malloc(k*sizeof(float));
    float *numX = (float*)malloc(k*sizeof(float));
    float *numY = (float*)malloc(k*sizeof(float));
    float *meanCentroidsX = (float*)malloc(k*sizeof(float));
    float *meanCentroidsY = (float*)malloc(k*sizeof(float));
    float oldCentroidsX[k];
    float oldCentroidsY[k];

    float difference = 0;
    int vote = 0;
    bool done = false;

    while(!done)
    {
        // 2a. calculate each distance
        for(size_t i = 0; i < n_data; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                distData[j] = dist(centroids[j], data[i]);
                distIdx[j] = j;
            }

            // 2b. labelling data depends on it's distance
            bubbleSort(distData, distIdx, k);
            data[i].label = distIdx[0];
            // printf("label = %d", distIdx[0]);
            // getchar();
        
            // 3. calculate mean of each centroids
            // 3a. sum all data each clusters
            for(int jj = 0; jj < k; ++jj)
            {
                if(data[i].label == jj)
                {
                    sumX[jj] += data[i].x;
                    sumY[jj] += data[i].y;

                    numX[jj]++;
                    numY[jj]++;
                    break;
                }
            }
        }

        // 3b. calculate mean
        for(int jj = 0; jj < k; ++jj)
        {
            meanCentroidsX[jj] = sumX[jj]/numX[jj];
            meanCentroidsY[jj] = sumY[jj]/numY[jj];
            // 4. determine new centroids
            centroids[jj] = {meanCentroidsX[jj], meanCentroidsY[jj], jj};
        }

        for(int jj = 0; jj < k; ++jj)
        {
            difference = oldCentroidsX[jj] - centroids[jj].x;
            if(absf(difference) < 0.00001)
            {
                vote++;
            }
            difference = oldCentroidsY[jj] - centroids[jj].y;
            if(absf(difference) < 0.00001)
            {
                vote++;
            }
        }


        printPt(centroids, k);
        printf("\n");
        getchar();

        if (vote == (k*2))
            done = true;
        else
            vote = 0;

        for(int jj = 0; jj < k; ++jj)
        {
            oldCentroidsX[jj] = centroids[jj].x;
            oldCentroidsY[jj] = centroids[jj].y;
        }
    }

    free(distData);
    free(distIdx);
    free(sumX);
    free(sumY);
    free(numX);
    free(numY);
    free(meanCentroidsX);
    free(meanCentroidsY);
}

int main (int argc, char** argv)
{
    int n_data = 10;
    int k = 2; //number of cluster

    Point *samples = new Point[n_data];
    Point *centroids = new Point[k];

    samples[0] = {1.4, 2.6, 0};
    samples[1] = {3.1, 3.4, 0};
    samples[2] = {10.4, 6.3, 0};
    samples[3] = {3.7, 2.8, 0};
    samples[4] = {6.9, 7.5, 0};
    samples[5] = {7.5, 8.9, 0};
    samples[6] = {23.8, 10, 0};
    samples[7] = {5.2, 4.9, 0};
    samples[8] = {7.2, 8.3, 0};
    samples[9] = {8.4, 7.7, 0};

    // 1. init centroids
    for(int i = 0; i < k; ++i)
    {
        centroids[i] = {(float)i*10, (float)i*10, 0};
        centroids[i].label = i;
    }
    
    // // 1. init centroids
    // for(int i = 0; i < k; ++i)
    // {
    //     centroids[i] = samples[i*5+3];
    //     centroids[i].label = i;
    // }

    printPt(centroids, k);
    kmeans(k, samples, n_data, centroids);

    printPt(samples, n_data);

    return 0;
}