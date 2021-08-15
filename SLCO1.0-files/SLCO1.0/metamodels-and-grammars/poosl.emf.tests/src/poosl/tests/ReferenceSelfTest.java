/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.PooslFactory;
import poosl.ReferenceSelf;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Reference Self</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ReferenceSelfTest extends ExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ReferenceSelfTest.class);
	}

	/**
	 * Constructs a new Reference Self test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ReferenceSelfTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Reference Self test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected ReferenceSelf getFixture() {
		return (ReferenceSelf)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createReferenceSelf());
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

} //ReferenceSelfTest
