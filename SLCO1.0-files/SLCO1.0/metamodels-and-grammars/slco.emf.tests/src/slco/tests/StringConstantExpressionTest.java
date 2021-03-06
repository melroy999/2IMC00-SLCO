/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.SlcoFactory;
import slco.StringConstantExpression;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>String Constant Expression</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class StringConstantExpressionTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(StringConstantExpressionTest.class);
	}

	/**
	 * Constructs a new String Constant Expression test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public StringConstantExpressionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this String Constant Expression test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected StringConstantExpression getFixture() {
		return (StringConstantExpression)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createStringConstantExpression());
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

} //StringConstantExpressionTest
