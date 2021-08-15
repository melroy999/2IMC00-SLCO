/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.IntegerConstantExpression;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Integer Constant Expression</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class IntegerConstantExpressionTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(IntegerConstantExpressionTest.class);
	}

	/**
	 * Constructs a new Integer Constant Expression test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IntegerConstantExpressionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Integer Constant Expression test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected IntegerConstantExpression getFixture() {
		return (IntegerConstantExpression)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createIntegerConstantExpression());
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

} //IntegerConstantExpressionTest
