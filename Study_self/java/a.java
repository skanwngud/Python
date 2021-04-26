class Circle {

final static double PI = 3.14;

public double radius;

public Circle(double radius) {

this.radius = radius;

}

public double getArea() {

return this.radius * this.radius * PI;

}

}

class Cylinder {

private Circle c;

private double height;

public Cylinder(Circle c, double h) {

this.c = c;

this.height = h;

}

public double getVolume() {

return c.getArea() * height;

}

}

public class project {

public static void main(String[] args) {

Circle c = new Circle(5.6);

Cylinder cy = new Cylinder(c, 2.7);

System.out.println("체적: " + cy.getVolume());

}

}