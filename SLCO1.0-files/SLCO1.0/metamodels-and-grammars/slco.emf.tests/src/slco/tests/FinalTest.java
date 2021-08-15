/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.Final;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Final</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class FinalTest extends VertexTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(FinalTest.class);
	}

	/**
	 * Constructs a new Final test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public FinalTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Final test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected Final getFixture() {
		return (Final)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createFinal());
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

} //FinalTest
