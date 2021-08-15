/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.IntegerConstant;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Integer Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class IntegerConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(IntegerConstantTest.class);
	}

	/**
	 * Constructs a new Integer Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IntegerConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Integer Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected IntegerConstant getFixture() {
		return (IntegerConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createIntegerConstant());
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

} //IntegerConstantTest
