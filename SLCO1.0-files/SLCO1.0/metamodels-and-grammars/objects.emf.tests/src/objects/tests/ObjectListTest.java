/**
 */
package objects.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import objects.ObjectList;
import objects.ObjectsFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Object List</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ObjectListTest extends TestCase {

	/**
	 * The fixture for this Object List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ObjectList fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ObjectListTest.class);
	}

	/**
	 * Constructs a new Object List test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ObjectListTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Object List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(ObjectList fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Object List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ObjectList getFixture() {
		return fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(ObjectsFactory.eINSTANCE.createObjectList());
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

} //ObjectListTest
