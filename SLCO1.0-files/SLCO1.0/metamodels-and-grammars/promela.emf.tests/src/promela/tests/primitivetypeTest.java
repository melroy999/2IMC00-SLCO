/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.primitivetype;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>primitivetype</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class primitivetypeTest extends typenameTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(primitivetypeTest.class);
	}

	/**
	 * Constructs a new primitivetype test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public primitivetypeTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this primitivetype test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected primitivetype getFixture() {
		return (primitivetype)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createprimitivetype());
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

} //primitivetypeTest
