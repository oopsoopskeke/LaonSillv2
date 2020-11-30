/**
 * @file Examples.h
 * @date 2017-06-16
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef EXAMPLES_H
#define EXAMPLES_H 

#define EXAMPLE_ITEM_DEF_NAME_MAXLEN   (256)
#define EXAMPLE_ITEM_DEF_DESC_MAXLEN   (256)

typedef void(*CBExampleFunc)();

typedef struct ExampleItemDef_s {
   char         name[EXAMPLE_ITEM_DEF_NAME_MAXLEN];
   char         desc[EXAMPLE_ITEM_DEF_DESC_MAXLEN];
   CBExampleFunc   func; 
} ExampleItemDef;

class Examples {
public: 
    Examples() {}
    virtual ~Examples() {}

    static void checkItem(const char* itemName);
    static void run(const char* itemName);
private:
    static void runItem(ExampleItemDef itemDef);

};
#endif /* EXAMPLES_H */
