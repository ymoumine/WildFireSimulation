#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <GL/glut.h>
#include <vector>
#include <random>    // Pour std::shuffle et std::mt19937 // For std::shuffle and std::mt19937
#include <algorithm> // Pour std::shuffle // For std::shuffle
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Define CHECK macro for error checking
#define CHECK(call)                                                   \
{                                                                     \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                         \
    {                                                                 \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                  \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                      \
    }                                                                 \
}


#define N 1000  // Taille de la grille // Grid size
#define BURN_DURATION 5000  // Dur�e de combustion d'un arbre en millisecondes (5 secondes) // Tree burning duration in milliseconds (5 seconds)
#define FIRE_START_COUNT 100 

// Utilisation de vecteurs pour gérer la mémoire // Using vectors to manage memory
// Use 1D vectors to manage memory
std::vector<int> forest(N* N, 0);
std::vector<int> burnTime(N* N, 0);
std::vector<int> newForest(N* N, 0);

// gpu memory
int* M_forest, * M_burnTime, * M_newForest;
dim3 grid, block;
int* d_allBurnedOut;
int n;

int simulationDuration = 60000;  // Durée de la simulation (60 secondes) // Simulation duration (60 seconds)
int startTime = 0;  // Temps de départ en millisecondes // Start time in milliseconds
int elapsedTime = 0;  // Temps écoulé // Elapsed time
float spreadProbability = 0.3f;  // Probabilité que le feu se propage à un arbre voisin // Probability that fire spreads to a neighboring tree

bool isPaused = false;  // Indicateur de pause // Pause indicator
int pauseStartTime = 0;  // Temps de début de la pause // Start time of pause

float zoomLevel = 1.0f;  // Niveau de zoom // Zoom level
float offsetX = 0.0f, offsetY = 0.0f;  // Décalage horizontal et vertical pour le déplacement // Horizontal and vertical offset for movement
float moveSpeed = 0.05f;  // Vitesse de déplacement de la vue // View movement speed

bool allBurnedOut = true;
bool dragging = false;  // Indicateur de glisser-déposer avec la souris // Mouse drag indicator
int lastMouseX, lastMouseY;  // Dernière position de la souris lors du clic // Last mouse position when clicked

// Fonction pour initialiser la forêt // Function to initialize the forest
void initializeForest() {
    // Initialisation de la forêt avec 50% d'arbres // Initializing the forest with 50% trees
    for (int i = 0; i < N * N; i++) {
        forest[i] = rand() % 2;  // 50% trees (1), 50% empty space (0)
        burnTime[i] = 0;  // No tree is burning at the start
        newForest[i] = forest[i];/// initilize new forest
    }

    // Liste de positions disponibles pour allumer les feux // List of available positions to start fires
    std::vector<int> availablePositions;
    for (int i = 0; i < N * N; i++) {
        if (forest[i] == 1) {  // Add positions with trees to the list
            availablePositions.push_back(i);
        }
    }

    // Mélanger les positions disponibles pour une distribution plus uniforme // Shuffle the available positions for a more uniform distribution
    std::random_device rd;  // Générateur de nombres aléatoires basé sur l'implémentation du système // Random number generator based on system implementation
    std::mt19937 g(rd());   // Générateur de nombres pseudo-aléatoires basé sur Mersenne Twister // Mersenne Twister-based pseudo-random number generator
    std::shuffle(availablePositions.begin(), availablePositions.end(), g);

    // Allumer des feux de manière uniforme sur la grille // Ignite fires uniformly across the grid
    for (int fire = 0; fire < FIRE_START_COUNT && fire < availablePositions.size(); fire++) {
        int idx = availablePositions[fire];
        forest[idx] = 2;  // Ignite the tree
        burnTime[idx] = BURN_DURATION;  // Set the burn duration
        
    }

    startTime = glutGet(GLUT_ELAPSED_TIME);  // Réinitialiser le temps de départ // Reset start time
    elapsedTime = 0;  // Réinitialiser le temps écoulé // Reset elapsed time
    isPaused = false;  // Fin de la pause // End of pause


    // Copy data to device
    CHECK(cudaMemcpy(M_forest, forest.data(), N * N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(M_newForest, newForest.data(), N * N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(M_burnTime, burnTime.data(), N * N * sizeof(int), cudaMemcpyHostToDevice));
}

// Fonction d'initialisation OpenGL // OpenGL initialization function
 void initGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);  // Couleur de fond blanche // White background color
    glEnable(GL_DEPTH_TEST);  // Activer le test de profondeur // Enable depth test
}

// Fonction pour dessiner la grille // Function to draw the grid
void drawForest() {
    float cellSize = 2.0f / N;  // Taille de chaque cellule ajustée par la taille N // Adjusted cell size based on grid size N

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Choisir la couleur en fonction de l'état de la cellule // Set color based on the state of the cell
            int idx = i * N + j;
            
            if (forest[idx] == 0) {
                glColor3f(0.8f, 0.8f, 0.8f);  // Empty space (gray)
            }
            else if (forest[idx] == 1) {
                glColor3f(0.0f, 1.0f, 0.0f);  // Tree (green)
            }
            else if (forest[idx] == 2) {
                glColor3f(1.0f, 0.0f, 0.0f);  // Tree on fire (red)
            }
            else if (forest[idx] == 3) {
                glColor3f(0.0f, 0.0f, 0.0f);  // Burned tree (black)
            }

            // Dessiner la cellule // Draw the cell
            float x = -1.0f + j * cellSize;
            float y = -1.0f + i * cellSize;
            glBegin(GL_QUADS);
            glVertex2f(x, y);
            glVertex2f(x + cellSize, y);
            glVertex2f(x + cellSize, y + cellSize);
            glVertex2f(x, y + cellSize);
            glEnd();
        }
    }
}

// Fonction d'affichage // Display function
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Effacer le tampon de couleur et de profondeur // Clear color and depth buffer
    glLoadIdentity();  // Réinitialiser la matrice modèle-vue // Reset the model-view matrix
    glTranslatef(offsetX, offsetY, 0.0f);  // Appliquer le décalage // Apply translation offset
    glScalef(zoomLevel, zoomLevel, 1.0f);  // Appliquer le zoom // Apply zoom
    drawForest();  // Dessiner la forêt // Draw the forest
    glutSwapBuffers();  // Échanger les tampons pour afficher l'image // Swap buffers to display the image
}



// Gestion du clavier pour zoomer/dézoomer et réinitialiser // Keyboard handling for zooming and resetting
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case '+':
        zoomLevel *= 1.1f;  // Augmenter le niveau de zoom // Increase zoom level
        break;
    case '-':
        zoomLevel /= 1.1f;  // Diminuer le niveau de zoom // Decrease zoom level
        if (zoomLevel < 0.1f) zoomLevel = 0.1f;
        break;
    case 'r':  // Touche pour réinitialiser // Reset key
        zoomLevel = 1.0f;  // Réinitialiser le zoom et le décalage // Reset zoom and offset
        offsetX = 0.0f;
        offsetY = 0.0f;
        break;
    case 27:  // Touche Échap pour quitter // Escape key to quit
        exit(0);
    }
    glutPostRedisplay();  // Redessiner la scène // Redraw the scene
}

// Gestion des touches fléchées pour déplacer la vue // Arrow keys handling for moving the view
void specialKeys(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_UP:
        offsetY += moveSpeed / zoomLevel;  // Déplacer la vue vers le haut // Move the view up
        break;
    case GLUT_KEY_DOWN:
        offsetY -= moveSpeed / zoomLevel;  // Déplacer la vue vers le bas // Move the view down
        break;
    case GLUT_KEY_LEFT:
        offsetX += moveSpeed / zoomLevel;  // Déplacer la vue vers la gauche // Move the view left
        break;
    case GLUT_KEY_RIGHT:
        offsetX -= moveSpeed / zoomLevel;  // Déplacer la vue vers la droite // Move the view right
        break;
    }
    glutPostRedisplay();  // Redessiner la scène // Redraw the scene
}

// Gestion de la souris pour déplacer la vue // Mouse handling for moving the view
void mouseMotion(int x, int y) {
    if (dragging) {
        offsetX += (x - lastMouseX) * moveSpeed / zoomLevel;  // Mettre à jour le décalage horizontal // Update horizontal offset
        offsetY -= (y - lastMouseY) * moveSpeed / zoomLevel;  // Mettre à jour le décalage vertical // Update vertical offset
        lastMouseX = x;
        lastMouseY = y;
        glutPostRedisplay();  // Redessiner la scène // Redraw the scene
    }
}

// Fonction pour gérer le clic de souris // Function to handle mouse clicks
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {  // Si le bouton gauche de la souris est enfoncé // If the left mouse button is pressed
        if (state == GLUT_DOWN) {
            dragging = true;
            lastMouseX = x;
            lastMouseY = y;
        }
        else {
            dragging = false;
        }
    }
}

// Kernel Function to update the forest and fire propagation
__global__ void updateForestOnGPUMix(int* M_forest, int* M_newForest, int* M_burnTime, int n, int* d_allBurnedOut)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // 2d to 1d
    unsigned int idx = iy * n + ix;
    unsigned int upindx = (iy - 1) * n + ix; // upper cell index
    unsigned int downindx = (iy + 1) * n + ix; // lower cell index
    unsigned int leftindx = iy * n + (ix - 1); // left cell index
    unsigned int rightindx = iy * n + (ix + 1); // right cell index


    //allBurnedOut = true;  // Indicateur pour vérifier si tous les feux sont éteints // Flag to check if all fires are out
    
    if (ix < n && iy < n) {

        M_newForest[idx] = M_forest[idx];

        //printf("Kernel running: ix = %d, iy = %d, idx = %d\n", ix, iy, idx);
        
        if (M_forest[idx] == 2) {  // Si l'arbre est en feu // If the tree is on fire
            M_burnTime[idx] -= 200;  // Réduire le temps de combustion // Reduce the burning time

            
            // Vérifier si le feu est éteint // Check if the fire is out
            if (M_burnTime[idx] <= 0) {
                M_newForest[idx] = 3;  // Marquer l'arbre comme brûlé // Mark the tree as burned
                
            }
            else {

                // Create a random state for each thread
                curandState state;
                curand_init(clock64(), idx, 0, &state);

                // Propagation du feu aux voisins // Propagation of fire to neighbors
                if (ix > 0 && M_forest[leftindx] == 1 && (curand_uniform(&state) < 0.3f)) {
                    M_newForest[leftindx] = 2;
                    M_burnTime[leftindx] = BURN_DURATION;
                }
                if (ix < N - 1 && M_forest[rightindx] == 1 && (curand_uniform(&state) < 0.3f)) {
                    M_newForest[rightindx] = 2;
                    M_burnTime[rightindx] = BURN_DURATION;
                }
                if (iy > 0 && M_forest[upindx] == 1 && (curand_uniform(&state) < 0.3f)) {
                    M_newForest[upindx] = 2;
                    M_burnTime[upindx] = BURN_DURATION;
                }
                if (iy < N - 1 && M_forest[downindx] == 1 && (curand_uniform(&state) < 0.3f)) {
                    M_newForest[downindx] = 2;
                    M_burnTime[downindx] = BURN_DURATION;
                }
            }

        }

        // Check if a tree is still burning
        if (M_forest[idx] == 2) {
            atomicExch(d_allBurnedOut, 0);  // Mark that not all trees are burned out
        }


        // Update the forest with the new state
        //M_forest[idx] = M_newForest[idx];

    }

}

// Define seconds function for timing
double seconds() {
    return static_cast<double>(clock()) / CLOCKS_PER_SEC;
}

// Fonction pour animer la simulation // Function to animate the simulation
void update(int value) {

    if (isPaused) {  // Si la simulation est en pause, réinitialiser la forêt après la pause // If the simulation is paused, reset the forest after the pause
        if (glutGet(GLUT_ELAPSED_TIME) - pauseStartTime >= 3000) {
            initializeForest();  // Réinitialiser la forêt après 3 secondes // Reset the forest after 3 seconds
        }
        return;
    }

    // Reset allBurnedOut to 1 before kernel launch
    int h_allBurnedOut = 1;
    CHECK(cudaMemcpy(d_allBurnedOut, &h_allBurnedOut, sizeof(int), cudaMemcpyHostToDevice));


    // performance timer
    double iStart = seconds();
    double iElaps = seconds() - iStart;
    iStart = seconds();


    updateForestOnGPUMix << < grid, block >> > (M_forest, M_newForest, M_burnTime, n, d_allBurnedOut);  // Mettre à jour la forêt à chaque cycle // Update the forest at each cycle
    

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;
    printf("updateForestOnGPUMix <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
        grid.y,
        block.x, block.y, iElaps);
    // Swap pointers after kernel execution
    int* temp = M_forest;
    M_forest = M_newForest;
    M_newForest = temp;

    // Copy the updated forest data back to host
    CHECK(cudaMemcpy(forest.data(), M_forest, N * N * sizeof(int), cudaMemcpyDeviceToHost));

    // Check if all trees are burned out
    CHECK(cudaMemcpy(&h_allBurnedOut, d_allBurnedOut, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_allBurnedOut == 1) {
        isPaused = true;
        pauseStartTime = glutGet(GLUT_ELAPSED_TIME);
    }

    glutPostRedisplay();  // Demander un nouveau rendu // Request a new rendering
    glutTimerFunc(200, update, 0);  // Programmer la prochaine mise à jour dans 200 ms // Schedule the next update in 200 ms
}

// Fonction principale // Main function
int main(int argc, char** argv) {

    //float* forest, * h_B, * hostRef, * gpuRef;
    //h_A = (float*)malloc(nBytes);
    //h_B = (float*)malloc(nBytes);


    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    srand(static_cast<unsigned>(time(NULL)));  // Initialiser le générateur de nombres aléatoires // Initialize random number generator
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Simulation de feux de forêt/Forest Fire Simulation");  // Créer la fenêtre OpenGL // Create the OpenGL window

    initGL();

    // set up data size of matrix
    n = N;

    int nxy = n * n;
    //matrix size in bits
    int nBytes = nxy * sizeof(int);

    // device flag for burnedout
    CHECK(cudaMalloc((void**)&d_allBurnedOut, sizeof(int)));

    // malloc device global memory
    CHECK(cudaMalloc((void**)&M_forest, nBytes));
    CHECK(cudaMalloc((void**)&M_burnTime, nBytes));
    CHECK(cudaMalloc((void**)&M_newForest, nBytes));

    //initialize data at host
    initializeForest();

    // invoke kernel at host side
    int nbrThreads = 16;

    block = dim3(nbrThreads, nbrThreads);
    grid = dim3((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    //iStart = seconds();
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    glutTimerFunc(200, update, 0);

    glutMainLoop();
   
    //iElaps = seconds() - iStart;
    //printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
    //    grid.y,
    //    block.x, block.y, iElaps);
    // 
    //// check kernel error

    //// no need to copy kernel result back to host side

    //// free device global memory
    CHECK(cudaFree(M_forest));
    CHECK(cudaFree(M_burnTime));
    CHECK(cudaFree(M_newForest));
    CHECK(cudaFree(d_allBurnedOut));

    //// reset device
    CHECK(cudaDeviceReset());


    
    return 0;
}
