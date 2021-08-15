/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.ParameterExpression;
import poosl.PooslFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Parameter Expression</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ParameterExpressionTest extends ExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ParameterExpressionTest.class);
	}

	/**
	 * Constructs a new Parameter Expression test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ParameterExpressionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Parameter Expression test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected ParameterExpression getFixture() {
		return (ParameterExpression)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createParameterExpression());
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

} //ParameterExpressionTest
