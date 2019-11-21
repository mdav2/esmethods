#include <string>
#include <bitset>
#define MAXNORB 500

class determinant
{
    private:
        std::bitset<MAXNORB> val;
        int sz;
        
    public:
        determinant(int ival) {
        //should have some sort of error checking here
            this->val = ival;
        }
        
        int operator - ( determinant * other ) {
            std::bitset<MAXNORB> temp = this->val^other->val;
            int ct = temp.count();
            return ct;
        }
};
