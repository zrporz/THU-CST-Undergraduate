#include <stdio.h>
#include <stdlib.h>

void attack_function() {
    printf("Attack success!\n");
}
void vulnerable_function() {
    char buffer[64];
    printf("input: ");
    gets(buffer);  // 这是一个不安全的函数
    printf("output: %s\n", buffer);
}

int main() {
    vulnerable_function();
    return 0;
}