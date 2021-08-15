/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.BinaryOperatorExpression;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Binary Operator Expression</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class BinaryOperatorExpressionTest extends ExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(BinaryOperatorExpressionTest.class);
	}

	/**
	 * Constructs a new Binary Operator Expression test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryOperatorExpressionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Binary Operator Expression test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected BinaryOperatorExpression getFixture() {
		return (BinaryOperatorExpression)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createBinaryOperatorExpression());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //BinaryOperatorExpressionTest
