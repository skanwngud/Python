package Study.java;

public class Assignment {

    public static void main(String[] args) {

        int a,b,c;
        c=0;
        for(c=2;c<=9;c+=3) {
        for(a=1;a<=9;++a) {
            for(b=c;b<c+3;++b) {
                System.out.printf("%d x %d = %d\t", b, a, a*b);
                }
                System.out.println("\n");
            }
            System.out.println("\n");
        }
    }        

}
