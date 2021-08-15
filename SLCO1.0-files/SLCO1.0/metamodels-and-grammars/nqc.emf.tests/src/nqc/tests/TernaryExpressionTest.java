/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.TernaryExpression;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Ternary Expression</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class TernaryExpressionTest extends CompoundExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(TernaryExpressionTest.class);
	}

	/**
	 * Constructs a new Ternary Expression test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TernaryExpressionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Ternary Expression test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected TernaryExpression getFixture() {
		return (TernaryExpression)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createTernaryExpression());
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

} //TernaryExpressionTest
