/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.SlcoFactory;
import slco.VariableExpression;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Variable Expression</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class VariableExpressionTest extends ExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(VariableExpressionTest.class);
	}

	/**
	 * Constructs a new Variable Expression test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public VariableExpressionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Variable Expression test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected VariableExpression getFixture() {
		return (VariableExpression)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createVariableExpression());
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

} //VariableExpressionTest
